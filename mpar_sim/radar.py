import time
import warnings
from mpar_sim.beam import Beam, RectangularBeam, beam_broadening_factor, beamwidth2aperture
from mpar_sim.common.coordinate_transform import sph2cart
from mpar_sim.platforms import Platform
from mpar_sim.reports import DetectionReport
from mpar_sim.radar_detection_generator import RadarDetectionGenerator
import numpy as np
from typing import Callable, List, Optional, Tuple, Union
from scipy import constants
from mpar_sim.job import Job, RadarJob


class Radar():
  """
  A multi-function phased array radar system

  Assumes a uniform rectangular array

  Parameters
    ----------
    n_elements_x : int, optional
        Number of horizontal array elements, by default 16
    n_element_y : int, optional
        Number of vertical array elements, by default 16
    element_spacing : float, optional
        Element spacing in wavelengths, by default 0.5
    element_tx_power : float, optional
        Peak transmit power per element, by default 10
    center_frequency : float, optional
        Transmit frequency, by default 3e9
    system_temperature : float, optional
        System operating temperature in Kelvin, by default 290
    noise_figure : float, optional
        System noise figure, by default 4
    beam_type : Callable, optional
        Subarray beam pattern shape, by default RectangularBeam
    azimuth_limits : np.ndarray, optional
        Azimuth scan limts of the array, by default np.array([-45, 45])
    elevation_limits : np.ndarray, optional
        Elevation scan limits of the array, by default np.array([-45, 45])
    false_alarm_rate : float, optional
        Probability of false alarms, by default 1e-6
    has_azimuth : bool, optional
        If true, detection report contains target azimuth information, by default True
    has_elevation : bool, optional
        If true, detection report contains target elevation information, by default True
    has_velocity : bool, optional
        If true, detection report contains target velocity information, by default True
    has_measurement_noise : bool, optional
        If true, radar measurements are corrupted by noise, by default False
        TODO: CURRENTLY DOES NOTHING
    has_false_alarms : bool, optional
        If true, false alarms are added to the detection reports, by default True
    has_range_ambiguities : bool, optional
        If true, range-ambiguous targets are aliased into the unambiguous limits of the system, by default True
    has_velocity_ambiguities : bool, optional
        If true, velocity-ambiguous targets are aliased into the unambiguous limits of the system, by default True
    has_scan_loss : bool, optional
        If true, the beam object accounts for changes in beamwidth and directivity due to scanning off boresight, by default True
  """

  def __init__(self,
               # Array parameters
               n_elements_x: int = 16,
               n_element_y: int = 16,
               element_spacing: float = 0.5,
               element_tx_power: float = 10,
               # System parameters
               center_frequency: float = 3e9,
               system_temperature: float = 290,
               noise_figure: float = 4,
               # Scan settings
               beam_type: Callable = RectangularBeam,
               azimuth_limits: np.ndarray = np.array([-45, 45]),
               elevation_limits: np.ndarray = np.array([-45, 45]),
               # Detection settings
               false_alarm_rate: float = 1e-6,
               # Data generator settings
               has_azimuth: bool = True,
               has_elevation: bool = True,
               has_velocity: bool = True,
               has_measurement_noise: bool = False,
               has_false_alarms: bool = True,
               has_range_ambiguities: bool = True,
               has_velocity_ambiguities: bool = True,
               has_scan_loss: bool = True,
               ) -> None:
    # Array parameters
    self.n_elements_x = n_elements_x
    self.n_elements_y = n_element_y
    self.element_spacing = element_spacing
    self.element_tx_power = element_tx_power
    # A matrix containing the data generator ID associated with each array element. If the element is currently unallocated, the value
    self.subarray_indices = np.ones(
        (n_elements_x, n_element_y), dtype=int)*-1

    # Radar system parameters
    self.center_frequency = center_frequency
    self.wavelength = constants.c / self.center_frequency
    self.element_spacing = element_spacing
    self.system_temperature = system_temperature
    self.noise_figure = noise_figure

    # Scan settings
    self.beam_type = beam_type
    self.azimuth_limits = azimuth_limits
    self.elevation_limits = elevation_limits

    # Create a detection generator object with the given settings
    self.data_generator = RadarDetectionGenerator(
        false_alarm_rate=false_alarm_rate,
        has_azimuth=has_azimuth,
        has_elevation=has_elevation,
        has_velocity=has_velocity,
        has_measurement_noise=has_measurement_noise,
        has_false_alarms=has_false_alarms,
        has_range_ambiguities=has_range_ambiguities,
        has_velocity_ambiguities=has_velocity_ambiguities,
        has_scan_loss=has_scan_loss,
    )

  def load_job(self, job: Job) -> None:
    """
    Allocate resources for the given radar job

    Parameters
    ----------
    job: Job
      The radar job to be scheduled and executed. The following parameters must be present in the job object:
        - bandwidth
        - pulsewidth
        - prf
        - n_pulses
        - azimuth_beamwidth
        - elevation_beamwidth
        - azimuth_steering_angle
        - elevation_steering_angle
    """

    # Handle input errors
    if job.azimuth_steering_angle < min(self.azimuth_limits) or job. azimuth_steering_angle > max(self.azimuth_limits):
      warnings.warn(
          "Azimuth steering angle is outside the azimuth limits of the radar")

    if job.elevation_steering_angle < min(self.elevation_limits) or job.elevation_steering_angle > max(self.elevation_limits):
      warnings.warn(
          "Elevation steering angle is outside the elevation limits of the radar")

    # Compute range/velocity resolutions
    range_resolution = constants.c / (2 * job.bandwidth)
    velocity_resolution = (self.wavelength / 2) * (job.prf / job.n_pulses)

    # Compute ambiguity limits
    max_unambiguous_range = constants.c / (2 * job.prf)
    max_unambiguous_radial_speed = (self.wavelength / 2) * (job.prf / 2)

    # Compute required subarray resources from the beamwidth
    aperture_length_x, aperture_length_y = tuple(beamwidth2aperture(
        np.array([job.azimuth_beamwidth, job.elevation_beamwidth]), self.wavelength))
    # Compute the number of elements in each dimension for this aperture size
    n_elements_x = int(
        np.ceil(aperture_length_x / (self.element_spacing*self.wavelength)))
    n_elements_y = int(
        np.ceil(aperture_length_y / (self.element_spacing*self.wavelength)))
    n_elements_total = n_elements_x * n_elements_y

    # Try to allocate the subarray
    if n_elements_total <= np.sum(self.subarray_indices == -1):
        self.subarray_indices[:n_elements_x,
                            :n_elements_y] = self.data_generator.id
    else:
        print(warnings.warn
        ("Attempted to allocate a subarray larger than the available aperture"))
        return 

    # Create a new beam from the parameter
    beam = self.beam_type(
        azimuth_beamwidth=job.azimuth_beamwidth,
        elevation_beamwidth=job.elevation_beamwidth,
        azimuth_steering_angle=job.azimuth_steering_angle,
        elevation_steering_angle=job.elevation_steering_angle,
        has_scan_loss=self.data_generator.has_scan_loss
    )

    # Compute power and gain for the beam
    tx_power = n_elements_total * self.element_tx_power
    array_gain = beam.gain
    pulse_compression_gain = job.bandwidth * job.pulsewidth
    noise_power = constants.Boltzmann * self.system_temperature * \
        self.noise_figure * job.bandwidth * n_elements_total
    loop_gain = job.n_pulses * pulse_compression_gain * tx_power * \
        array_gain**2 * self.wavelength**2 / ((4*np.pi)**3 * noise_power)

    # Assign results to data generator
    self.data_generator.n_pulses = job.n_pulses
    self.data_generator.max_unambiguous_range = max_unambiguous_range
    self.data_generator.max_unambiguous_radial_speed = max_unambiguous_radial_speed
    self.data_generator.range_resolution = range_resolution
    self.data_generator.velocity_resolution = velocity_resolution
    self.data_generator.loop_gain = loop_gain
    self.data_generator.beam = beam
    # TODO: For now, assuming no AoA estimation is performed so the az/el resolution is just the beamwidth
    self.data_generator.azimuth_resolution = beam.azimuth_beamwidth
    self.data_generator.elevation_resolution = beam.elevation_beamwidth

  def detect(self, targets: List[Platform], time: float) -> DetectionReport:
      # TODO: For now, assumes only one beam can be formed at a time
    return self.data_generator.detect(targets, time)

  def reset_subarrays(self) -> None:
      """
      Reset subarray assignments
      """
      self.subarray_indices[:,:] = -1 
