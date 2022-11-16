
from typing import List

import numpy as np
from scipy import special

from mpar_sim.beam.beam import beam_scan_loss
from mpar_sim.common.coordinate_transform import azel2rotmat, cart2sph
from mpar_sim.common.wrap_to_interval import wrap_to_interval
from mpar_sim.old.platforms import Platform
from mpar_sim.old.reports import DetectionReport
from mpar_sim.common.albersheim import albersheim_pd


class RadarDetectionGenerator():
  """
  A detection-level model of a phased array radar system that generates detections using the radar range equation.

  """
  id = 0

  def __init__(self,
               # Detection settings
               false_alarm_rate: float = 1e-6,
               min_detection_snr_db: float = 0,
               # Detection report specifications
               has_azimuth: bool = True,
               has_elevation: bool = True,
               has_velocity: bool = True,
               has_measurement_noise: bool = False,
               has_false_alarms: bool = True,
               has_range_ambiguities: bool = True,
               has_velocity_ambiguities: bool = True,
               has_scan_loss: bool = True,
               ) -> None:
    ##############################
    # Input Parameters
    ##############################

    # Detection settings
    self.false_alarm_probability = false_alarm_rate
    self.min_detection_snr_db = min_detection_snr_db

    # Detection report specifications
    self.has_azimuth = has_azimuth
    self.has_elevation = has_elevation
    self.has_velocity = has_velocity
    self.has_measurement_noise = has_measurement_noise
    self.has_false_alarms = has_false_alarms
    self.has_range_ambiguities = has_range_ambiguities
    self.has_velocity_ambiguities = has_velocity_ambiguities
    self.has_scan_loss = has_scan_loss

    # Give each radar a unique ID
    self.id = RadarDetectionGenerator.id
    RadarDetectionGenerator.id += 1

  def detect(self, targets: List[Platform], time: float) -> DetectionReport:
    """
    Generate detections for the list of targets

    Args:
        targets (List[Platform]): List of targets to detect
    """

    if not isinstance(targets, list):
      targets = [targets]

    if len(targets) == 0:
      # No targets, return empty report
      return DetectionReport(
          time=time,
          range_resolution=self.range_resolution,
          velocity_resolution=self.velocity_resolution,
          azimuth_resolution=self.beam.azimuth_beamwidth,
          elevation_resolution=self.beam.elevation_beamwidth,
      )

    # Transform target coordinates into radar observation frame
    # TODO: For now, assume that the radar is at the origin, so only the look angle matters
    R = azel2rotmat(self.beam.azimuth_steering_angle,
                    self.beam.elevation_steering_angle)
    target_positions = R.T @ np.array(
        [target.position for target in targets]).T
    target_velocities = np.array([target.velocity for target in targets]).T
    target_rcs = np.array([target.rcs for target in targets])

    # Generate spherical observations from truth data
    [target_azimuths_rad, target_elevations_rad, target_ranges] = cart2sph(
        target_positions[0, :], target_positions[1, :], target_positions[2, :])
    target_azimuths = np.rad2deg(target_azimuths_rad)
    target_elevations = np.rad2deg(target_elevations_rad)

    # Range rate is the dot product of the target velocity and the radar look vector towards the target
    target_directions = target_positions / target_ranges
    target_radial_velocities = np.sum(
        target_velocities * target_directions, axis=0)

    # Wrap range measurements into the unambiguous range limits
    target_ranges = wrap_to_interval(
        target_ranges, 0, self.max_unambiguous_range)

    # Wrap measurements into the unambiguous velocity limits
    if self.has_velocity and self.has_velocity_ambiguities:
      target_radial_velocities = wrap_to_interval(
          target_radial_velocities, -self.max_unambiguous_radial_speed, self.max_unambiguous_radial_speed)

    # Compute the SNR from the radar range equation
    # The loop gain is the part of the radar range equation that doesn't depend on the target
    snr_db = 10*np.log10(self.loop_gain) + \
        10*np.log10(target_rcs) - 40*np.log10(target_ranges)

    # Compute losses due to beam shape and target angle
    beam_shape_loss_db = self.beam.shape_loss(
        target_azimuths, target_elevations)

    # Compute losses due to steering off boresight
    # TODO: This should already be accounted for as a loss in directivity
    # array_scan_loss = beam_scan_loss(
    #     self.beam.azimuth_steering_angle, self.beam.elevation_steering_angle)

    # Compute the instantaneous power, accounting for the beam shape
    # snr_db = snr_db - \
        # beam_shape_loss_db - array_scan_loss
    snr_db = snr_db - beam_shape_loss_db    

    # Compute probability of detection from the mean SNR
    snr = 10**(snr_db/10)
    pfa = self.false_alarm_probability
    N = self.n_pulses
    # Single pulse probability of detection
    # TODO: Assuming swerling 0 for now
    detection_probabilities = albersheim_pd(snr, pfa, N)

    # Report detections in a standardized format
    # TODO: Since no AoA processing is used, the azimuth and elevation angles that are reported are the steering angle of the antenna. In the local radar coordinate frame, this is (az,el) = (0,0).
    report = DetectionReport(
        rng=np.asarray(target_ranges),
        azimuth=np.zeros_like(target_ranges),
        elevation=np.zeros_like(target_ranges),
        velocity=np.asarray(target_radial_velocities),
        time=time,
        azimuth_resolution=self.beam.azimuth_beamwidth,
        elevation_resolution=self.beam.elevation_beamwidth,
        range_resolution=self.range_resolution,
        velocity_resolution=self.velocity_resolution,
        detection_probability=detection_probabilities,
        snr=snr_db,
        target_ids=[target.id for target in targets],
    )

    # TODO: Add noise to the measurements for tracking

    # Add false alarms
    report = self.generate_false_alarms(report)

    # Merge detections that cannot be resolved by the radar
    report.merge()

    # Account for non-unity probability of detection
    # This is only necessary for non-false alarm targets, since for false alarms the detection probability is just the false alarm probability
    if report.detection_probability is not None:
      true_target_indices = np.where(report.target_ids != -1)[0]
      is_detected = np.random.rand(
          len(true_target_indices)) < report.detection_probability[true_target_indices]
      # Also remove targets if their SNR falls below the pre-defined threshold
      is_detected = np.logical_and(is_detected,
                                   report.snr[true_target_indices] > self.min_detection_snr_db)

      # Delete targets that were not detected
      report.remove(true_target_indices[~is_detected])

    # Remove quantities that the radar does not measure
    if not self.has_azimuth:
      report.azimuth = None
      report.azimuth_beamwidth = None
    if not self.has_elevation:
      report.elevation = None
      report.elevation_beamwidth = None
    if not self.has_velocity:
      report.velocity = None
      report.velocity_resolution = None

    return report

  def generate_false_alarms(self, report: DetectionReport = DetectionReport()) -> DetectionReport:
    """
    Generate measurements corresponding to false alarms

    Returns:
          np.ndarray: Array of false alarm measurements
    """
    # Compute the number of resolution cells in each dimension that is being measured
    n_cells_total = self.count_resolution_cells()

    n_expected_false_alarms = n_cells_total * self.false_alarm_probability

    # If the expected number of false alarms is not an integer, treat the fractional part as the probability of an additional false alarm.
    fractional_false_alarms = n_expected_false_alarms % 1
    n_expected_false_alarms = int(
        n_expected_false_alarms - fractional_false_alarms)
    n_false_alarms = n_expected_false_alarms + \
        (np.random.rand() < fractional_false_alarms)

    # Use a NP decision rule to set a threshold for detecting a signal with a given probability of false alarm
    threshold_db = 20 * \
        np.log10(np.sqrt(special.gammaincinv(
            1, 1-self.false_alarm_probability)))

    # Uniformly distribute the false alarms across resolution cells
    range = np.random.rand(n_false_alarms) * self.max_unambiguous_range
    az = np.random.rand(
        n_false_alarms) * self.beam.azimuth_beamwidth - self.beam.azimuth_beamwidth/2
    el = np.random.rand(
        n_false_alarms) * self.beam.elevation_beamwidth - self.beam.elevation_beamwidth/2
    velocity = np.random.rand(
        n_false_alarms) * 2*self.max_unambiguous_radial_speed - self.max_unambiguous_radial_speed
    false_alarm_ids = np.ones(n_false_alarms, dtype=int) * -1
    detection_probabilities = np.ones(
        n_false_alarms) * self.false_alarm_probability
    snr = np.ones(n_false_alarms) * threshold_db

    # Append false alarm measurements to existing report
    report.range = np.concatenate((report.range, range))
    report.azimuth = np.concatenate(
        (report.azimuth, az))
    report.elevation = np.concatenate(
        (report.elevation, el))
    report.velocity = np.concatenate(
        (report.velocity, velocity))
    report.target_ids = np.concatenate((report.target_ids, false_alarm_ids))
    report.detection_probability = np.concatenate(
        (report.detection_probability, detection_probabilities))
    report.snr = np.concatenate((report.snr, snr))

    return report

  def count_resolution_cells(self) -> int:
    """
    Compute the total number of resolution cells in the radar's field of view
    Returns:
        int: Total number of cells
    """

    # Range cells
    n_range_cells = int(self.max_unambiguous_range / self.range_resolution)

    # Azimuth cells
    if self.has_azimuth:
      n_az_cells = int(self.beam.azimuth_beamwidth / self.azimuth_resolution)
    else:
      n_az_cells = 1

    # Elevation cells
    if self.has_elevation:
      n_el_cells = int(self.beam.elevation_beamwidth /
                       self.elevation_resolution)
    else:
      n_el_cells = 1

    # Range rate cells
    if self.has_velocity:
      n_velocity_cells = int(
          2*self.max_unambiguous_radial_speed / self.velocity_resolution)
    else:
      n_velocity_cells = 1

    # Total number of cells
    n_cells_total = n_az_cells * n_el_cells * n_range_cells * n_velocity_cells
    return n_cells_total
