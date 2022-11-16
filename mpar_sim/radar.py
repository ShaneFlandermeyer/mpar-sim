import copy
from typing import Callable, Set, Tuple, Union

import numpy as np
from scipy import constants
from stonesoup.base import Property
from stonesoup.models.measurement import MeasurementModel
from stonesoup.sensor.radar.radar import RadarElevationBearingRangeRate
from stonesoup.sensor.sensor import Sensor
from stonesoup.types.array import CovarianceMatrix, StateVector
from stonesoup.types.detection import TrueDetection
from stonesoup.types.groundtruth import GroundTruthState
from stonesoup.types.state import StateVector

from mpar_sim.beam.beam import RectangularBeam
from mpar_sim.beam.common import beam_broadening_factor
from mpar_sim.common.albersheim import albersheim_pd
from mpar_sim.common.coordinate_transform import (cart2sph, rotx, roty, rotz)
from mpar_sim.look import Look
from mpar_sim.models.measurement.estimation import (angle_crlb, range_crlb,
                                                    velocity_crlb)
from mpar_sim.models.measurement.nonlinear import RangeRangeRateBinningAliasing


class PhasedArrayRadar(Sensor):
  """An active electronically scanned array (AESA) radar sensor"""

  # Motion and orientation parameters
  rotation_offset: StateVector = Property(
      default=StateVector([0, 0, 0]),
      doc="A 3x1 array of angles (rad), specifying the radar orientation in terms of the "
      "counter-clockwise rotation around the :math:`x,y,z` axis. i.e Roll, Pitch and Yaw. "
      "Default is ``StateVector([0, 0, 0])``")
  position_mapping: Tuple[int, int, int] = Property(
      default=(0, 1, 2),
      doc="Mapping between or positions and state "
          "dimensions. [x,y,z]")
  measurement_model: MeasurementModel = Property(
      default=RadarElevationBearingRangeRate(
          position_mapping=(0, 2, 4),
          velocity_mapping=(1, 4, 5),
          noise_covar=np.array([0, 0, 0, 0])),
      doc="The measurement model used to generate "
      "measurements. By default, this object measures range, range rate, azimuth, and elevation with no noise."
  )
  # Array parameters
  n_elements_x: int = Property(
      default=16,
      doc="The number of horizontal array elements")
  n_elements_y: int = Property(
      default=16,
      doc="The number of vertical array elements")
  element_spacing: float = Property(
      default=0.5,
      doc="The spacing between array elements (m)")
  element_tx_power: float = Property(
      default=10,
      doc="Tx power of each element (W)")
  # System parameters
  center_frequency: float = Property(
      default=3e9,
      doc="Transmit frequency of the array")
  system_temperature: float = Property(
      default=290,
      doc="System noise temperature (K)")
  noise_figure: float = Property(
      default=4,
      doc="Receiver noise figure (dB)")
  # Scan settings
  beam_shape: Callable = Property(
      default=RectangularBeam,
      doc="Object describing the shape of the beam.")
  # Detections settings
  false_alarm_rate: float = Property(
      default=1e-6,
      doc="Probability of false alarm")
  max_range: float = Property(
      default=np.inf,
      doc="Maximum detection range of the radar (m)"
  )
  field_of_view: float = Property(
      default=90,
      doc="The width in each dimension for which targets can be detected (deg)."
  )

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.wavelength = constants.c / self.center_frequency
    # TODO: Handle subarray resource allocation in a separate object.

  @ measurement_model.getter
  def measurement_model(self):
    measurement_model = copy.deepcopy(self._property_measurement_model)
    measurement_model.translation_offset = self.position.copy()
    measurement_model.rotation_offset = self.rotation_offset.copy()
    return measurement_model

  @property
  def _rotation_matrix(self):
    """
    3D rotation matrix for converting target pointing vectors to the sensor frame

    Returns
    -------
    np.ndarray
      (3,3) 3D rotation matrix
    """
    theta_x = -np.deg2rad(self.rotation_offset[0, 0])  # Roll
    theta_y = -np.deg2rad(self.rotation_offset[1, 0])  # Pitch/elevation
    theta_z = -np.deg2rad(self.rotation_offset[2, 0])  # Yaw/azimuth

    return rotz(theta_z) @ roty(theta_y) @ rotx(theta_x)

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

    # Create a new beam from the parameter set
    az_broadening, el_broadening = beam_broadening_factor(
        look.azimuth_steering_angle,
        look.elevation_steering_angle)
    effective_az_beamwidth = look.azimuth_beamwidth * az_broadening
    effective_el_beamwidth = look.elevation_beamwidth * el_broadening
    self.beam = self.beam_shape(
        azimuth_beamwidth=effective_az_beamwidth,
        elevation_beamwidth=effective_el_beamwidth,
        azimuth_steering_angle=look.azimuth_steering_angle,
        elevation_steering_angle=look.elevation_steering_angle,
    )

    # Compute the loop gain (the part of the radar range equation that doesn't depend on the target)
    self.tx_power = look.tx_power
    pulse_compression_gain = look.bandwidth * look.pulsewidth
    n_elements_total = np.ceil(look.tx_power / self.element_tx_power)
    noise_power = constants.Boltzmann * self.system_temperature * \
        self.noise_figure * look.bandwidth * n_elements_total
    self.loop_gain = look.n_pulses * pulse_compression_gain * self.tx_power * \
        self.beam.gain**2 * self.wavelength**2 / ((4*np.pi)**3 * noise_power)

    self.measurement_model = RangeRangeRateBinningAliasing(
        range_res=self.range_resolution,
        range_rate_res=self.velocity_resolution,
        max_unambiguous_range=self.max_unambiguous_range,
        max_unambiguous_range_rate=self.max_unambiguous_radial_speed,
        ndim_state=6,
        mapping=[0, 2, 4],
        velocity_mapping=[1, 3, 5],
        noise_covar=CovarianceMatrix(np.diag([0, 0, 0, 0])))

  def is_detectable(self, state: GroundTruthState) -> bool:
    measurement_vector = self.measurement_model.function(state, noise=False)
    # Check if state falls within sensor's FOV
    fov_min = -self.field_of_view / 2
    fov_max = +self.field_of_view / 2
    az_t = measurement_vector[0, 0].degrees - self.beam.azimuth_steering_angle
    el_t = measurement_vector[1, 0].degrees - \
        self.beam.elevation_steering_angle
    true_range = measurement_vector[2, 0]
    return fov_min <= az_t <= fov_max and fov_min <= el_t <= fov_max and true_range <= self.max_range

  def measure(self, ground_truths: Set[GroundTruthState], noise: Union[np.ndarray, bool] = True, **kwargs) -> set[TrueDetection]:
    detections = set()
    measurement_model = copy.deepcopy(self.measurement_model)

    # Loop through the targets and generate detections
    for truth in ground_truths:
      # Skip targets that are not detectable
      if not self.is_detectable(truth):
        continue

      # Get the position of the target in the radar coordinate frame
      relative_pos = truth.state_vector[self.position_mapping,
                                        :] - self.position
      relative_pos = self._rotation_matrix @ relative_pos

      # Convert target position to spherical coordinates
      [target_az, target_el, r] = cart2sph(*relative_pos)

      # Compute target's az/el relative to the beam center
      relative_az = np.rad2deg(target_az) - self.beam.azimuth_steering_angle
      relative_el = np.rad2deg(target_el) - self.beam.elevation_steering_angle
      # Compute loss due to the target being off-centered in the beam
      beam_shape_loss_db = self.beam.shape_loss(relative_az, relative_el)

      snr_db = 10*np.log10(self.loop_gain) + 10*np.log10(truth.rcs) - \
          40*np.log10(r) - beam_shape_loss_db

      # Probability of detection
      if snr_db > 0:
        N = self.n_pulses
        pfa = self.false_alarm_rate
        pd = albersheim_pd(snr_db, pfa, N)
      else:
        pd = 0  # Assume targets are not detected with negative SNR

      # Add detections based on the probability of detection
      if np.random.rand() <= pd:

        # # Use the SNR to compute the measurement accuracies in each dimension. These accuracies are set to the CRLB of each quantity (i.e., we assume we have efficient estimators)
        snr_lin = 10**(snr_db/10)
        single_pulse_snr = snr_lin / self.n_pulses
        # The CRLB uses the RMS bandwidth. Assuming an LFM waveform with a rectangular spectrum, B_rms = B / sqrt(12)
        rms_bandwidth = self.bandwidth / np.sqrt(12)
        rms_range_res = constants.c / (2 * rms_bandwidth)
        range_variance = range_crlb(
            snr=single_pulse_snr,
            resolution=rms_range_res,
            bias_fraction=0.05)
        velocity_variance = velocity_crlb(
            snr=snr_lin,
            resolution=self.velocity_resolution,
            bias_fraction=0.05)
        azimuth_variance = angle_crlb(
            snr=snr_lin,
            resolution=self.beam.azimuth_beamwidth,
            bias_fraction=0.01)
        elevation_variance = angle_crlb(
            snr=snr_lin,
            resolution=self.beam.elevation_beamwidth,
            bias_fraction=0.01)
        measurement_model.noise_covar = np.diag(
            [elevation_variance, azimuth_variance, range_variance, velocity_variance])

        measurements = measurement_model.function(truth, noise=noise)
        detection = TrueDetection(measurements,
                                  timestamp=truth.timestamp,
                                  measurement_model=measurement_model,
                                  groundtruth_path=truth)
        detections.add(detection)
      # TODO: Add in some false alarms.

    return detections