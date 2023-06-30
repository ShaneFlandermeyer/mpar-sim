from datetime import datetime
from typing import Callable, List, Optional, Union

import numpy as np
from scipy import constants
from scipy.special import gammaincinv

from mpar_sim.beam.beam import RectangularBeam
from mpar_sim.beam.common import (aperture2beamwidth, beam_broadening_factor,
                                  beam_scan_loss, beamwidth2aperture)
from mpar_sim.common.coordinate_transform import cart2sph
from mpar_sim.models.measurement.base import MeasurementModel
from mpar_sim.models.measurement.estimation import (angle_crlb, range_crlb,
                                                    velocity_crlb)
from mpar_sim.models.measurement.nonlinear import CartesianToRangeAzElRangeRate
from mpar_sim.types.detection import Clutter, TrueDetection
from mpar_sim.types.groundtruth import GroundTruthState
from mpar_sim.types.look import Look
from mpar_sim.common.util import as_column_vec
import jax.numpy as jnp


class PhasedArrayRadar():
  """An active electronically scanned array (AESA) radar sensor"""

  def __init__(self,
               # Motion and orientation parameters
               ndim_state: int = 6,
               position: np.ndarray = np.zeros((3,)),
               velocity: np.ndarray = np.zeros((3,)),
               position_mapping: List[int] = [0, 2, 4],
               velocity_mapping: List[int] = [1, 3, 5],
               measurement_model: MeasurementModel = None,
               rotation_offset: np.ndarray = np.zeros((3, 1)),
               timestamp: datetime = None,
               # Phased array parameters
               n_elements_x: int = 16,
               n_elements_y: int = 16,
               element_spacing: float = 0.5,
               element_tx_power: float = 10,
               element_gain: float = 3,
               max_az_beamwidth: float = np.Inf,
               max_el_beamwidth: float = np.Inf,
               # System parameters
               center_frequency: float = 3e9,
               sample_rate: float = 100e6,
               system_temperature: float = 290,
               noise_figure: float = 4,
               # Scan settings
               beam_shape: Callable = RectangularBeam,
               # Detection settings
               pfa: float = 1e-6,
               alias_measurements: bool = False,
               discretize_measurements: bool = False,
               noise_covar: Optional[np.ndarray] = None,
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
    self.alias_measurements = alias_measurements
    self.discretize_measurements = discretize_measurements
    self.noise_covar = noise_covar

    # Compute the maximum possible beamwidth in az/el for the array geometry
    aperture_width = self.n_elements_x * self.element_spacing * self.wavelength
    aperture_height = self.n_elements_y * self.element_spacing * self.wavelength
    min_beamwidths = aperture2beamwidth(
        np.array([aperture_width, aperture_height]), self.wavelength)
    self.min_az_beamwidth = min_beamwidths[0]
    self.min_el_beamwidth = min_beamwidths[1]
    self.max_az_beamwidth = max_az_beamwidth
    self.max_el_beamwidth = max_el_beamwidth

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

    # Compute range/velocity resolutions
    self.range_resolution = constants.c / (2 * self.bandwidth)
    self.velocity_resolution = (
        self.wavelength / 2) * (self.prf / self.n_pulses)

    # Compute ambiguity limits
    self.max_unambiguous_range = constants.c / (2 * self.prf)
    self.max_unambiguous_radial_speed = (self.wavelength / 2) * (self.prf / 2)

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
      beamwidths = np.array([look.azimuth_beamwidth, look.elevation_beamwidth])
      tx_aperture_size = beamwidth2aperture(
          beamwidths, self.wavelength) / self.wavelength
      n_tx_elements = np.prod(np.ceil(
          tx_aperture_size / self.element_spacing).astype(int))
      self.tx_power = n_tx_elements * self.element_tx_power

    # Compute the loop gain, which is the portion of the SNR computation that does not depend on the target RCS or range.
    pulse_compression_gain = look.bandwidth * look.pulsewidth
    noise_power = constants.Boltzmann * self.system_temperature * \
        10**(self.noise_figure/10) * self.sample_rate
    self.loop_gain = look.n_pulses * self.tx_power * pulse_compression_gain * \
        10**(self.element_gain/10) * self.tx_beam.gain * self.rx_beam.gain * \
        self.wavelength**2 / ((4*np.pi)**3 * noise_power)

    self.measurement_model = CartesianToRangeAzElRangeRate(
        range_res=self.range_resolution,
        range_rate_res=self.velocity_resolution,
        max_unambiguous_range=self.max_unambiguous_range,
        max_unambiguous_range_rate=self.max_unambiguous_radial_speed,
        position_mapping=self.position_mapping,
        velocity_mapping=self.velocity_mapping,
        alias_measurements=self.alias_measurements,
        discretize_measurements=self.discretize_measurements,
        noise_covar=self.noise_covar,
    )
    
  def measure(self, targets, noise, timestamp):
      pass

  def measure(self,
              ground_truths: List[GroundTruthState],
              noise: Union[np.ndarray, bool] = True,
              timestamp: Optional[datetime] = None,
              **kwargs) -> List[Union[TrueDetection, Clutter]]:
    """
    Generates stochastic detections from a set of target ground truths
    Parameters
    ----------
    ground_truths : List[GroundTruthState]
        True information of targets to be measured
    noise : Union[np.ndarray, bool], optional
        If true, noise is added to each detection. This noise includes aliasing, discretization into bins, and measurement accuracy limits dictated by the CRLB of the measurement error for each quantity, by default True
    Returns
    -------
    List[TrueDetection]
        Detections made by the radar
    """
    detections = []
    measurement_model = self.measurement_model
    measurement_model.translation_offset = np.array(self.position)
    measurement_model.rotation_offset = np.array(self.rotation_offset)

    n_targets = len(ground_truths)
    state_vectors = np.atleast_2d(
        np.array([truth.state_vector.ravel() for truth in ground_truths])).T
    rcs = np.array([truth[-1].rcs for truth in ground_truths])

    # Get the position of the target in the radar coordinate frame
    relative_pos = state_vectors[self.position_mapping, :] - self.position
    relative_pos = measurement_model.rotation_matrix @ relative_pos
    [target_az, target_el, r] = cart2sph(*relative_pos, degrees=True)
    relative_az = target_az - self.tx_beam.azimuth_steering_angle
    relative_el = target_el - self.tx_beam.elevation_steering_angle

    # Compute SNR and probability of detection assuming a Swerling 1 target model
    beam_shape_loss_db = self.tx_beam.shape_loss(relative_az, relative_el)
    beam_scan_loss_db = beam_scan_loss(self.tx_beam.azimuth_steering_angle,
                                       self.tx_beam.elevation_steering_angle)
    snr_db = 10*np.log10(self.loop_gain) + 10*np.log10(rcs) - \
        40*np.log10(r) - beam_shape_loss_db - beam_scan_loss_db
    snr_lin = 10**(snr_db/10)
    pfa = self.pfa
    pd = pfa**(1/(1+snr_lin))

    # Filter out targets that have low SNR, are outside the radar's FOV, or are outside the main beam, then determine which targets are detected
    pd[snr_db < 0] = 0
    pd[~self.is_detectable(target_az, target_el, r)] = 0
    pd[~self.is_in_beam(relative_az, relative_el)] = 0
    is_detected = np.random.uniform(0, 1, size=n_targets) < pd
    n_detections = np.count_nonzero(is_detected)

    if n_detections > 0:
      # Concatenate the measurement noise covariance matrices to pass them into the measurement model
      if noise:
        if self.noise_covar:
          measurement_model.noise_covar = self.noise_covar
        else:
          # If the noise covariance matrix is not manually provided, compute it from the CRLB. Since the CRLB depends on the SNR, we need to compute it separately for each target
          measurement_model.noise_covar = np.zeros(
              (measurement_model.ndim_meas, measurement_model.ndim_meas, n_detections))

          single_pulse_snr = snr_lin / self.n_pulses
          # The CRLB uses the RMS bandwidth. Assuming an LFM waveform with a rectangular spectrum, B_rms = B / sqrt(12)
          rms_bandwidth = self.bandwidth / np.sqrt(12)
          rms_range_res = constants.c / (2 * rms_bandwidth)
          azimuth_variance = angle_crlb(
              snr=snr_lin[is_detected],
              resolution=self.rx_beam.azimuth_beamwidth,
              bias_fraction=0.01)
          elevation_variance = angle_crlb(
              snr=snr_lin[is_detected],
              resolution=self.rx_beam.elevation_beamwidth,
              bias_fraction=0.01)
          range_variance = range_crlb(
              snr=single_pulse_snr[is_detected],
              resolution=rms_range_res,
              bias_fraction=0.05)
          velocity_variance = velocity_crlb(
              snr=snr_lin[is_detected],
              resolution=self.velocity_resolution,
              bias_fraction=0.05)
          for i in range(n_detections):
            measurement_model.noise_covar[..., i] = np.diag(
                [azimuth_variance[i],
                 elevation_variance[i],
                 range_variance[i],
                 velocity_variance[i]])
      measurements = measurement_model.function(
          state_vectors[:, is_detected], noise=noise)

      detection_inds = is_detected.nonzero()[0]
      for i in range(n_detections):
        detection = TrueDetection(state_vector=measurements[:, i],
                                  snr=snr_db[detection_inds[i]],
                                  timestamp=timestamp,
                                  measurement_model=measurement_model,
                                  groundtruth_path=ground_truths[detection_inds[i]])
        detections.append(detection)

    if self.include_false_alarms:
      # Generate uniformly distributed false alarms in the radar beam
      # Compute the number of false alarms
      n_range_bins = int(self.max_unambiguous_range / self.range_resolution)
      n_vel_bins = int(2*self.max_unambiguous_radial_speed /
                       self.velocity_resolution)
      n_expected_false_alarms = self.pfa * n_range_bins * n_vel_bins
      n_false_alarms = int(np.random.poisson(n_expected_false_alarms))

      # Generate random false alarm measurements
      az = np.random.uniform(low=-self.tx_beam.azimuth_beamwidth/2,
                             high=self.tx_beam.azimuth_beamwidth/2,
                             size=n_false_alarms) + self.tx_beam.azimuth_steering_angle
      el = np.random.uniform(low=-self.tx_beam.elevation_beamwidth/2,
                             high=self.tx_beam.elevation_beamwidth/2,
                             size=n_false_alarms) + self.tx_beam.elevation_steering_angle
      r = np.random.uniform(low=0,
                            high=self.max_unambiguous_range,
                            size=n_false_alarms)
      v = np.random.uniform(low=-self.max_unambiguous_radial_speed,
                            high=self.max_unambiguous_radial_speed,
                            size=n_false_alarms)

      # Bin range and velocity
      if self.discretize_measurements:
        r = np.floor(r / self.range_resolution) * \
            self.range_resolution + self.range_resolution/2
        v = np.floor(v / self.velocity_resolution) * \
            self.velocity_resolution + self.velocity_resolution/2

      # Add false alarms to the detection report
      snr_db = 20*np.log10(gammaincinv(1, 1 - self.pfa))
      for i in range(n_false_alarms):
        state_vector = np.array([[az[i]],
                                 [el[i]],
                                 [r[i]],
                                 [v[i]]])
        detection = Clutter(state_vector=state_vector,
                            snr=snr_db,
                            timestamp=timestamp if timestamp else None,
                            measurement_model=measurement_model)
        detections.append(detection)

    return detections

  def is_detectable(self,
                    target_az: float,
                    target_el: float,
                    target_range: float) -> bool:
    """
    Returns true if the target is within the radar's field of view (in range, azimuth, and elevation) and false otherwise
    Parameters
    ----------
    target_az: float
        Azimuth angle of the target in degrees
    target_el: float
        Elevation angle of the target in degrees
    target_range: float
        Range of the target in meters
    Returns
    -------
    bool
        Whether the target can be detected by the radar
    """
    valid_az = np.logical_and(
        self.az_fov[0] <= target_az, target_az <= self.az_fov[1])
    valid_el = np.logical_and(
        self.el_fov[0] <= target_el, target_el <= self.el_fov[1])
    valid_range = np.logical_and(
        self.min_range <= target_range, target_range <= self.max_range)
    return np.logical_and(valid_az, np.logical_and(valid_el, valid_range))

  def is_in_beam(self, relative_az: float, relative_el: float) -> bool:
    """
    Returns true if the target is within the radar's beam and false otherwise
    Parameters
    ----------
    target_az: float
        Azimuth angle of the target in degrees
    target_el: float
        Elevation angle of the target in degrees
    Returns
    -------
    bool
        Whether the target is within the radar's beam
    """
    return np.logical_and(
        np.abs(relative_az) <= 0.5 * self.tx_beam.azimuth_beamwidth,
        np.abs(relative_el) <= 0.5 * self.tx_beam.elevation_beamwidth)
