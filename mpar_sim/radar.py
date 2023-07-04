import math
import random
from datetime import datetime
from functools import cached_property
from typing import Callable, List, Optional, Union

import jax
from scipy.spatial.transform import Rotation
from scipy import constants
from scipy.special import gammaincinv

from mpar_sim.beam.beam import RectangularBeam
from mpar_sim.beam.common import (aperture2beamwidth, beam_broadening_factor,
                                  beam_scan_loss, beamwidth2aperture)
from mpar_sim.common.coordinate_transform import cart2sph
from mpar_sim.models.measurement.base import MeasurementModel
from mpar_sim.models.measurement.estimation import (angle_crlb, range_crlb,
                                                    velocity_crlb)
from mpar_sim.models.measurement.nonlinear import CartesianToRangeAzElVelocity
from mpar_sim.models.rcs import Swerling
from mpar_sim.models.transition.linear import ConstantVelocity
from mpar_sim.types.detection import Clutter, TrueDetection
from mpar_sim.types.look import Look
from mpar_sim.types.target import Target
import numpy as np


class PhasedArrayRadar():
  """An active electronically scanned array (AESA) radar sensor"""

  def __init__(self,
               # Motion and orientation parameters
               ndim_state: int = 6,
               position: np.ndarray = np.zeros((3,)),
               velocity: np.ndarray = np.zeros((3,)),
               rotation_offset: np.ndarray = np.zeros((3, 1)),
               position_mapping: List[int] = [0, 2, 4],
               velocity_mapping: List[int] = [1, 3, 5],
               measurement_model: MeasurementModel = None,
               timestamp: datetime = None,
               # Phased array parameters
               n_elements_x: int = 16,
               n_elements_y: int = 16,
               element_spacing: float = 0.5,
               element_tx_power: float = 10,
               element_gain: float = 3,
               # System parameters
               center_frequency: float = 3e9,
               sample_rate: float = 100e6,
               system_temperature: float = 290,
               noise_figure: float = 4,
               # Scan settings
               beam_shape: Callable = RectangularBeam,
               # Detection settings
               pfa: float = 1e-6,
               include_false_alarms: bool = False,
               alias_measurements: bool = False,
               discretize_measurements: bool = False,
               seed=random.randint(0, 2**32-1),
               ) -> None:
    self.ndim_state = ndim_state
    self.position = position
    self.velocity = velocity
    self.rotation_offset = rotation_offset
    self.position_mapping = position_mapping
    self.velocity_mapping = velocity_mapping
    self.measurement_model = measurement_model
    self.timestamp = timestamp
    self.n_elements_x = n_elements_x
    self.n_elements_y = n_elements_y
    self.element_spacing = element_spacing
    self.element_tx_power = element_tx_power
    self.element_gain = element_gain
    self.center_frequency = center_frequency
    self.wavelength = constants.c / self.center_frequency
    self.sample_rate = sample_rate
    self.system_temperature = system_temperature
    self.noise_figure = noise_figure
    self.beam_shape = beam_shape
    self.pfa = pfa
    self.include_false_alarms = include_false_alarms
    self.alias_measurements = alias_measurements
    self.discretize_measurements = discretize_measurements

    self.key = jax.random.PRNGKey(seed)

  @property
  def range_resolution(self):
    return constants.c / (2 * self.bandwidth)

  @property
  def velocity_resolution(self):
    return (self.wavelength / 2) * (self.prf / self.n_pulses)

  @property
  def unambiguous_range(self):
    return constants.c / (2 * self.prf)

  @property
  def unambiguous_velocity(self):
    return (self.wavelength / 2) * (self.prf / 2)

  @cached_property
  def min_az_beamwidth(self):
    aperture_width = self.n_elements_x * self.element_spacing * self.wavelength
    return aperture2beamwidth(aperture_width, self.wavelength)

  @cached_property
  def min_el_beamwidth(self):
    aperture_height = self.n_elements_y * self.element_spacing * self.wavelength
    return aperture2beamwidth(aperture_height, self.wavelength)

  @property
  def global_to_antenna(self):
    ypr = self.rotation_offset
    ypr[1] = -ypr[1]
    return Rotation.from_euler('zyx', ypr.ravel(), degrees=True).inv()

  def load_look(self, look: Look):
    """
    Allocate resources for the given radar job

    Parameters
    ----------
    look: Look
      The radar job to be scheduled and executed. The following parameters must be present in the job object:
        - bandwidth
        - pulsewidth
        - prf
        - n_pulses
        - azimuth_beamwidth
        - elevation_beamwidth
        - azimuth_steering_angle
        - elevation_steering_angle
        - n_elements_x
        - n_elements_y
    """
    # Waveform parameters
    self.bandwidth = look.bandwidth
    self.pulsewidth = look.pulsewidth
    self.prf = look.prf
    self.n_pulses = look.n_pulses

    # Create beams from the parameter set
    az_broadening, el_broadening = beam_broadening_factor(
        look.azimuth_steering_angle,
        look.elevation_steering_angle)
    tx_az_beamwidth = look.azimuth_beamwidth * az_broadening
    tx_el_beamwidth = look.elevation_beamwidth * el_broadening
    # Note: This object assumes that the beam is only spoiled on transmit, so that the receive beam uses the full aperture.
    rx_az_beamwidth = self.min_az_beamwidth * az_broadening
    rx_el_beamwidth = self.min_el_beamwidth * el_broadening
    self.tx_beam = self.beam_shape(
        wavelength=self.wavelength,
        azimuth_beamwidth=tx_az_beamwidth,
        elevation_beamwidth=tx_el_beamwidth,
        azimuth_steering_angle=look.azimuth_steering_angle,
        elevation_steering_angle=look.elevation_steering_angle,
    )
    self.rx_beam = self.beam_shape(
        wavelength=self.wavelength,
        azimuth_beamwidth=rx_az_beamwidth,
        elevation_beamwidth=rx_el_beamwidth,
        azimuth_steering_angle=look.azimuth_steering_angle,
        elevation_steering_angle=look.elevation_steering_angle,
    )

    # If the user specifies a transmit power, use that. Otherwise, compute the transmit power from the aperture size.
    if look.tx_power:
      self.tx_power = look.tx_power
    else:
      beamwidths = np.array(
          [look.azimuth_beamwidth, look.elevation_beamwidth])
      tx_aperture_size = beamwidth2aperture(
          beamwidths, self.wavelength) / self.wavelength
      n_tx_elements = np.prod(np.ceil(
          tx_aperture_size / self.element_spacing).astype(int))
      self.tx_power = n_tx_elements * self.element_tx_power

    # Compute the loop gain, which is the portion of the single-pulse SNR computation that does not depend on the target RCS or range.
    pulse_compression_gain = look.bandwidth * look.pulsewidth
    noise_power = constants.Boltzmann * self.system_temperature * \
        10**(self.noise_figure/10) * self.sample_rate
    self.loop_gain = self.tx_power * pulse_compression_gain * \
        10**(self.element_gain/10) * self.tx_beam.gain * self.rx_beam.gain * \
        self.wavelength**2 / ((4*np.pi)**3 * noise_power)

  def measure(self,
              targets: List[Target],
              noise: bool = False,
              timestamp: float = None):
    # Compute spherical target positions in the radar coordinate frame
    n_targets = len(targets)
    positions = np.array(
        [target.position for target in targets]).reshape((n_targets, -1))
    relative_pos = positions - self.position.reshape((1, -1))
    relative_pos = self.global_to_antenna.apply(relative_pos)
    [target_az, target_el, target_r] = cart2sph(*relative_pos.T, degrees=True)
    relative_az = target_az - self.tx_beam.azimuth_steering_angle
    relative_el = target_el - self.tx_beam.elevation_steering_angle

    # Compute SNR/probability of detection
    # TODO: This may vary on each pulse, depending on the target model
    rcs = np.array([target.rcs for target in targets])
    beam_shape_loss = self.tx_beam.shape_loss(relative_az, relative_el)
    scan_loss = beam_scan_loss(relative_az, relative_el)
    single_pulse_snr_db = 10*np.log10(self.loop_gain) + 10*np.log10(
        rcs) - 40*np.log10(target_r) - beam_shape_loss - scan_loss

    pd = np.empty((n_targets, ))
    for i, target in enumerate(targets):
      pd[i] = target.detection_probability(
          pfa=self.pfa, n_pulse=self.n_pulses, snr_db=single_pulse_snr_db[i])
    self.key, subkey = jax.random.split(self.key)
    is_detected = jax.random.uniform(subkey, shape=(n_targets,)) < pd
    n_detections = np.count_nonzero(is_detected)

    if n_detections > 0:
      positions = positions[is_detected]
      velocities = np.array([target.velocity for target in targets]).reshape(
          (n_targets, -1))[is_detected]
      if self.measurement_model is None:
        # Create the measurement models based on SNR and resolutions
        measurement_model = CartesianToRangeAzElVelocity(
            translation_offset=self.position,
            rotation_offset=self.rotation_offset,
            velocity=self.velocity,
            range_resolution=self.range_resolution,
            velocity_resolution=self.velocity_resolution,
            discretize_measurements=self.discretize_measurements,
            unambiguous_range=self.unambiguous_range,
            unambiguous_velocity=self.unambiguous_velocity,
            alias_measurements=self.alias_measurements,
        )
        pos_inds = np.array(measurement_model.position_mapping)
        vel_inds = np.array(measurement_model.velocity_mapping)
        state_vectors = np.zeros((self.ndim_state, n_detections))
        state_vectors[pos_inds] = positions.T
        state_vectors[vel_inds] = velocities.T
        measurements = np.empty((measurement_model.ndim_meas, n_detections))

        # Compute errors in each measurement dimension
        single_pulse_snr = 10**(single_pulse_snr_db/10)
        # Coherent integration SNR
        snr = single_pulse_snr * self.n_pulses
        snr_db = 10*np.log10(snr)
        # The CRLB uses the RMS bandwidth. Assuming an LFM waveform with a rectangular spectrum, B_rms = B / sqrt(12)
        rms_bandwidth = self.bandwidth / math.sqrt(12)
        rms_range_res = constants.c / (2 * rms_bandwidth)
        az_variance = angle_crlb(
            snr=snr[is_detected],
            resolution=self.rx_beam.azimuth_beamwidth,
            bias_fraction=0.01)
        el_variance = angle_crlb(
            snr=snr[is_detected],
            resolution=self.rx_beam.elevation_beamwidth,
            bias_fraction=0.01)
        range_variance = range_crlb(
            snr=single_pulse_snr[is_detected],
            resolution=rms_range_res,
            bias_fraction=0.05)
        vel_variance = velocity_crlb(
            snr=snr[is_detected],
            resolution=self.velocity_resolution,
            bias_fraction=0.05)

        detections = []
        detection_inds = is_detected.nonzero()[0]
        for i in range(n_detections):
          measurement_model.noise_covar = np.diag(
              np.array([az_variance[i],
                         el_variance[i],
                         range_variance[i],
                         vel_variance[i]]))
          measurements[:, i] = measurement_model(
            state_vectors[:, i], noise=noise)
      else:
        pos_inds = np.array(measurement_model.position_mapping)
        vel_inds = np.array(measurement_model.velocity_mapping)
        state_vectors = np.zeros((self.ndim_state, n_detections))
        state_vectors[pos_inds] = positions.T
        state_vectors[vel_inds] = velocities.T
        measurements = self.measurement_model(state_vectors, noise=noise)

    detections = []
    detection_inds = is_detected.nonzero()[0]
    for i in detection_inds:
      detection = TrueDetection(
          timestamp=self.timestamp,
          measurement=measurements[:, i],
          measurement_model=measurement_model,
          snr=snr_db[i],
      )
      detections.append(detection)

    if self.include_false_alarms:
      false_alarms = self._generate_false_alarms()
      detections.extend(false_alarms)

    return detections

  def _generate_false_alarms(self):
    """
    Generate uniformly distributed false alarms in the radar beam
    """
    # Compute the number of false alarms from a Poisson random process
    self.key, subkey = jax.random.split(self.key)
    n_range_bins = int(self.unambiguous_range / self.range_resolution)
    n_vel_bins = int(2*self.unambiguous_velocity /
                     self.velocity_resolution)
    n_expected_false_alarms = self.pfa * n_range_bins * n_vel_bins
    n_false_alarms = int(
        jax.random.poisson(subkey, lam=n_expected_false_alarms))
    if n_false_alarms == 0:
      return []

    # Generate random false alarm measurements
    self.key, *subkeys = jax.random.split(self.key, 5)
    az = jax.random.uniform(subkeys[0],
                            minval=-self.tx_beam.azimuth_beamwidth/2,
                            maxval=self.tx_beam.azimuth_beamwidth/2,
                            shape=(n_false_alarms,)) + self.tx_beam.azimuth_steering_angle
    el = jax.random.uniform(subkeys[1],
                            minval=-self.tx_beam.elevation_beamwidth/2,
                            maxval=self.tx_beam.elevation_beamwidth/2,
                            shape=(n_false_alarms,)) + self.tx_beam.elevation_steering_angle
    r = jax.random.uniform(subkeys[2],
                           minval=0,
                           maxval=self.unambiguous_range,
                           shape=(n_false_alarms,))
    v = jax.random.uniform(subkeys[3],
                           minval=-self.unambiguous_velocity,
                           maxval=self.unambiguous_velocity,
                           shape=(n_false_alarms,))

    # Discretize false alarm measurements
    if self.discretize_measurements:
      r = np.floor(r / self.range_resolution) * \
          self.range_resolution + self.range_resolution/2
      v = np.floor(v / self.velocity_resolution) * \
          self.velocity_resolution + self.velocity_resolution/2

    # Add false alarms to the detection list
    snr_db = 10*np.log10(gammaincinv(1, 1-self.pfa))
    false_alarms = []
    for i in range(n_false_alarms):
      measurement = np.array([az[i], el[i], r[i], v[i]])
      detection = Clutter(measurement=measurement,
                          snr=snr_db,
                          timestamp=self.timestamp)
      false_alarms.append(detection)

    return false_alarms


if __name__ == '__main__':
  radar = PhasedArrayRadar(
      include_false_alarms=False,
      alias_measurements=True,
      discretize_measurements=True,
  )
  look = Look(
      azimuth_beamwidth=55,
      elevation_beamwidth=55,
      azimuth_steering_angle=0,
      elevation_steering_angle=0,
      center_frequency=1e9,
      bandwidth=100e6,
      pulsewidth=10e-6,
      prf=5e3,
      n_pulses=100e3)
  radar.load_look(look)
  targets = [Target(position=[100, 0, 50],
                    velocity=[100, 0, 0],
                    transition_model=ConstantVelocity(
      ndim_pos=3, noise_diff_coeff=1),
      rcs=Swerling(case=0, mean=1)) for _ in range(1)]
  detections = radar.measure(targets)
  print(detections[0].snr)
  detections = radar.measure(targets)
  print(detections[0].snr)