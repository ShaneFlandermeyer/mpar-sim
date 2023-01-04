import copy
from functools import lru_cache
from typing import Callable, List, Optional, Set, Tuple, Union

import numpy as np
from scipy import constants
from stonesoup.base import Property
from stonesoup.models.measurement import MeasurementModel
from stonesoup.sensor.radar.radar import RadarElevationBearingRangeRate
from stonesoup.sensor.sensor import Sensor
from stonesoup.types.array import StateVector
from stonesoup.types.detection import TrueDetection
from stonesoup.types.groundtruth import GroundTruthState
from stonesoup.types.state import StateVector
from stonesoup.types.detection import Clutter

from mpar_sim.beam.beam import RectangularBeam
from mpar_sim.beam.common import beam_broadening_factor
from mpar_sim.common.coordinate_transform import (
    cart2sph, rotx, roty, rotz)
from mpar_sim.looks.look import Look
from mpar_sim.looks.spoiled_look import SpoiledLook
from mpar_sim.models.measurement.estimation import (angle_crlb, range_crlb,
                                                    velocity_crlb)
from mpar_sim.models.measurement.nonlinear import RangeRangeRateBinningAliasing
from mpar_sim.beam.common import aperture2beamwidth
from datetime import datetime
from stonesoup.base import clearable_cached_property


class PhasedArrayRadar(Sensor):
  """An active electronically scanned array (AESA) radar sensor"""

  # Motion and orientation parameters
  ndim_state: int = Property(
      default=6,
      doc="Number of state dimensions for the target.")
  position_mapping: Tuple[int, int, int] = Property(
      default=(0, 2, 4),
      doc="Mapping between or positions and state "
          "dimensions. [x,y,z]")
  velocity_mapping: Tuple[int, int, int] = Property(
      default=(1, 3, 5),
      doc="Mapping between velocity components and state "
      "dimensions. [vx,vy,vz]")
  measurement_model: MeasurementModel = Property(
      default=RadarElevationBearingRangeRate(
          ndim_state=6,
          position_mapping=(0, 2, 4),
          velocity_mapping=(1, 4, 5),
          noise_covar=np.array([0, 0, 0, 0])),
      doc="The measurement model used to generate "
      "measurements. By default, this object measures range, range rate, azimuth, and elevation with no noise.")
  rotation_offset: StateVector = Property(
      default=StateVector([0, 0, 0]),
      doc="A 3x1 array of angles (rad), specifying the radar orientation in terms of the "
      "counter-clockwise rotation around the :math:`x,y,z` axis. i.e Roll, Pitch and Yaw. "
      "Default is ``StateVector([0, 0, 0])``")
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
  include_false_alarms: bool = Property(
      default=True,
      doc="Whether to include false alarms in the detections")
  max_range: float = Property(
      default=np.inf,
      doc="Maximum detection range of the radar (m). If a target is beyond this range, it will never be detected.")
  az_fov: Union[List, np.ndarray] = Property(
      default=np.array([-90, 90]),
      doc="Azimuth slice within which the radar can detect targets (deg). The first element in the array is the lower bound, the second is the upper bound.")
  el_fov: Union[List, np.ndarray] = Property(
      default=np.array([-90, 90]),
      doc="Elevation slice within which the radar can detect targets (deg). The first element in the array is the lower bound, the second is the upper bound.")

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.wavelength = constants.c / self.center_frequency

    # Compute the maximum possible beamwidth in az/el for the array geometry
    aperture_width = self.n_elements_x * self.element_spacing * self.wavelength
    aperture_height = self.n_elements_y * self.element_spacing * self.wavelength
    beamwidths = aperture2beamwidth(
        np.array([aperture_width, aperture_height]), self.wavelength)
    self.max_az_beamwidth = beamwidths[0]
    self.max_el_beamwidth = beamwidths[1]

  @measurement_model.getter
  def measurement_model(self):
    measurement_model = copy.deepcopy(self._property_measurement_model)
    measurement_model.translation_offset = np.array(self.position)
    measurement_model.rotation_offset = np.array(self.rotation_offset)
    return measurement_model

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
    # If the transmit beam is spoiled, use the full aperture to form the receive beam. Otherwise, use the requested beamwidth for both transmit and receive
    if isinstance(look, SpoiledLook):
      rx_az_beamwidth = self.max_az_beamwidth * az_broadening
      rx_el_beamwidth = self.max_el_beamwidth * el_broadening
    else:
      rx_az_beamwidth = tx_az_beamwidth
      rx_el_beamwidth = tx_el_beamwidth
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

    # Compute the loop gain (the part of the radar range equation that doesn't depend on the target)
    self.tx_power = look.tx_power
    pulse_compression_gain = look.bandwidth * look.pulsewidth
    n_elements_total = np.ceil(look.tx_power / self.element_tx_power)
    noise_power = constants.Boltzmann * self.system_temperature * \
        self.noise_figure * look.bandwidth * n_elements_total
    self.loop_gain = look.n_pulses * pulse_compression_gain * self.tx_power * \
        self.tx_beam.gain * self.rx_beam.gain * \
        self.wavelength**2 / ((4*np.pi)**3 * noise_power)

    self.measurement_model = RangeRangeRateBinningAliasing(
        range_res=self.range_resolution,
        range_rate_res=self.velocity_resolution,
        max_unambiguous_range=self.max_unambiguous_range,
        max_unambiguous_range_rate=self.max_unambiguous_radial_speed,
        ndim_state=self.ndim_state,
        mapping=self.position_mapping,
        velocity_mapping=self.velocity_mapping,
        noise_covar=np.diag([0.1, 0.1, 0.1, 0.1]))

  @lru_cache
  def is_detectable(self, 
                    target_az: float, target_el: float, target_range: float) -> bool:
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
    return (self.az_fov[0] <= target_az <= self.az_fov[1]) and \
           (self.el_fov[0] <= target_el <= self.el_fov[1]) and \
           (target_range <= self.max_range)
           
  @clearable_cached_property('rotation_offset')
  def _rotation_matrix(self) -> np.ndarray:
    """3D axis rotation matrix"""
    theta_x = -self.rotation_offset[0, 0]  # roll
    theta_y = self.rotation_offset[1, 0]  # pitch#elevation
    theta_z = -self.rotation_offset[2, 0]  # yaw#azimuth
    return rotz(theta_z) @ roty(theta_y) @ rotx(theta_x)

  def measure(self,
              ground_truths: Set[GroundTruthState],
              noise: Union[np.ndarray, bool] = True,
              timestamp: Optional[datetime] = None,
              **kwargs) -> set[TrueDetection, Clutter]:
    """
    Generates stochastic detections from a set of target ground truths
    Parameters
    ----------
    ground_truths : Set[GroundTruthState]
        True information of targets to be measured
    noise : Union[np.ndarray, bool], optional
        If true, noise is added to each detection. This noise includes aliasing, discretization into bins, and measurement accuracy limits dictated by the CRLB of the measurement error for each quantity, by default True
    Returns
    -------
    set[TrueDetection]
        Detections made by the radar
    """
    detections = set()
    measurement_model = self.measurement_model
    # Compute the rotation matrix that maps the global coordinate frame into the radar frame
    
    # Loop through the targets and generate detections
    for truth in ground_truths:
      # Get the position of the target in the radar coordinate frame
      relative_pos = truth.state_vector[self.position_mapping,
                                        :] - self.position
      relative_pos = self._rotation_matrix @ relative_pos

      # Convert target position to spherical coordinates
      [target_az, target_el, r] = cart2sph(*relative_pos)
      
      # Skip targets that are not detectable
      if not self.is_detectable(target_az, target_el, r):
          continue

      # Compute target's az/el relative to the beam center
      relative_az = np.rad2deg(target_az) - self.tx_beam.azimuth_steering_angle
      relative_el = np.rad2deg(target_el) - \
          self.tx_beam.elevation_steering_angle
      # Compute loss due to the target being off-centered in the beam
      beam_shape_loss_db = self.tx_beam.shape_loss(relative_az, relative_el)

      snr_db = 10*np.log10(self.loop_gain) + 10*np.log10(truth.rcs) - \
          40*np.log10(r) - beam_shape_loss_db
      snr_lin = 10**(snr_db/10)

      # Probability of detection
      if snr_db > 0:
        pfa = self.false_alarm_rate
        pd = pfa**(1/(1+snr_lin))
      else:
        pd = 0  # Assume targets are not detected with negative SNR

      # Add detections based on the probability of detection
      if np.random.rand() <= pd:

        # # Use the SNR to compute the measurement accuracies in each dimension. These accuracies are set to the CRLB of each quantity (i.e., we assume we have efficient estimators)
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
            resolution=self.rx_beam.azimuth_beamwidth,
            bias_fraction=0.01)
        elevation_variance = angle_crlb(
            snr=snr_lin,
            resolution=self.rx_beam.elevation_beamwidth,
            bias_fraction=0.01)
        measurement_model.noise_covar = np.diag(
            [elevation_variance,
             azimuth_variance,
             range_variance,
             velocity_variance])

        measurements = measurement_model.function(truth, noise=noise)
        detection = TrueDetection(measurements,
                                  timestamp=truth.timestamp,
                                  measurement_model=measurement_model,
                                  groundtruth_path=truth)
        detections.add(detection)

    if self.include_false_alarms:
      # Generate uniformly distributed false alarms in the radar beam
      # Compute the number of false alarms
      n_range_bins = int(self.max_unambiguous_range / self.range_resolution)
      n_vel_bins = int(2*self.max_unambiguous_radial_speed /
                       self.velocity_resolution)
      n_expected_false_alarms = self.false_alarm_rate * n_range_bins * n_vel_bins
      n_false_alarms = int(np.random.poisson(n_expected_false_alarms))

      # Generate random false alarm positions
      el = np.random.uniform(low=-self.tx_beam.elevation_beamwidth/2,
                             high=self.tx_beam.elevation_beamwidth/2,
                             size=n_false_alarms) + self.tx_beam.elevation_steering_angle
      az = np.random.uniform(low=-self.tx_beam.azimuth_beamwidth/2,
                             high=self.tx_beam.azimuth_beamwidth/2,
                             size=n_false_alarms) + self.tx_beam.azimuth_steering_angle
      r = np.random.uniform(low=0,
                            high=self.max_unambiguous_range,
                            size=n_false_alarms)
      v = np.random.uniform(low=-self.max_unambiguous_radial_speed,
                            high=self.max_unambiguous_radial_speed,
                            size=n_false_alarms)

      # Bin range and velocity
      r = np.floor(r / self.range_resolution) * \
          self.range_resolution + self.range_resolution/2
      v = np.floor(v / self.velocity_resolution) * \
          self.velocity_resolution + self.velocity_resolution/2

      # Add false alarms to the detection report
      for i in range(n_false_alarms):
        detections.add(Clutter(np.array([[np.deg2rad(el[i])],
                                         [np.deg2rad(az[i])],
                                         [r[i]],
                                         [v[i]]]),
                               timestamp=truth.timestamp if truth else timestamp,
                               measurement_model=measurement_model))

    return detections
