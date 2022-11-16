from copy import deepcopy
from typing import List
import numpy as np
from mpar_sim.common.coordinate_transform import azel2rotmat
from mpar_sim.look import RadarLook

from mpar_sim.old.platforms import Platform
from mpar_sim.radar import Radar
from scipy import constants
import matplotlib.pyplot as plt

from mpar_sim.old.reports import DetectionReport


class RadarEnvironment():
  def __init__(self,
               targets: List[Platform],
               radar: Radar,
               observation_width: int = 128,
               observation_height: int = 128,
               render=True
               ):
    self.targets = targets
    self.radar = radar
    self.observation_width = observation_width
    self.observation_height = observation_height

    self.global_time = 0
    self.initial_target_state = deepcopy(targets)
    self.initial_radar_state = deepcopy(radar)
    self.beam_coverage_map = np.zeros((observation_width, observation_height))
    self.beam_coverage_epsilon = 0.9

    self.last_observation = None
    self.fig = None

  def step(self, action: RadarLook):
    self.radar.reset_subarrays()
    self.radar.load_look(action)

    # Collect measurments
    detections = self.radar.detect(self.targets, self.global_time)

    # Convert detections to a resampled range-doppler map
    range_vel_detection_map = self.create_range_velocity_map(action, detections)

    # Update the beam coverage map
    self.update_beam_coverage(action)

    observation = np.stack([range_vel_detection_map, self.beam_coverage_map], axis=2)

    self.last_observation = observation
    self.last_info = detections

    # Update scenario
    for target in self.targets:
      target.update(dt=action.dwell_time)

    self.global_time += action.dwell_time

    return observation, detections

  def render(self):
    if self.last_observation is None:
      return

    if self.fig is None:
      self.fig = plt.figure()
      self.ax = self.fig.gca()
      self.fig.show()

    self.ax.clear()
    # Plot the beam coverage map
    self.ax.imshow(self.last_observation[..., 1],
                   extent=(np.min(self.az_axis), np.max(self.az_axis),
                           np.min(self.el_axis), np.max(self.el_axis)),
                   aspect='auto')
    self.ax.set_xlabel('Azimuth')
    self.ax.set_ylabel('Elevation')
    self.fig.canvas.draw()
    plt.pause(0.01)

  def reset(self):
    self.radar = deepcopy(self.initial_radar_state)
    self.targets = deepcopy(self.initial_target_state)
    self.global_time = 0

  def close(self):
    if self.render:
      plt.close()

  def update_beam_coverage(self, action: RadarLook) -> None:
    az_beamwidth = self.radar.data_generator.beam.azimuth_beamwidth
    el_beamwidth = self.radar.data_generator.beam.elevation_beamwidth
    az_axis = np.linspace(
        self.radar.azimuth_limits[0] - az_beamwidth/2,
        self.radar.azimuth_limits[1] + az_beamwidth/2,
        self.observation_width)
    el_axis = np.linspace(
        self.radar.elevation_limits[0] - el_beamwidth/2,
        self.radar.elevation_limits[1] + el_beamwidth/2, self.observation_height)
    # Y-axis of the image increases with increasing elevation, so we need to flip the axis above
    el_axis = np.flip(el_axis)

    az_steering_angle = action.azimuth_steering_angle
    el_steering_angle = action.elevation_steering_angle
    az_beam_indices = np.where(
        (az_axis >= az_steering_angle - az_beamwidth / 2) & (az_axis <= az_steering_angle + az_beamwidth / 2))[0]
    el_beam_indices = np.where(
        (el_axis >= el_steering_angle - el_beamwidth / 2) & (el_axis <= el_steering_angle + el_beamwidth / 2))[0]
    az_grid, el_grid = np.meshgrid(az_beam_indices, el_beam_indices)
    # Decay the beam coverage map pixels by a small amount
    self.beam_coverage_map *= self.beam_coverage_epsilon
    # Reset the beam pixels that were just covered to 1
    self.beam_coverage_map[el_grid, az_grid] = 1
    self.az_axis = az_axis
    self.el_axis = el_axis

  def create_range_velocity_map(self, 
                                action: RadarLook, 
                                detections: DetectionReport) -> np.ndarray:
    prf = action.prf
    pri = 1 / prf
    max_unambiguous_range = constants.c * pri / 2
    max_unambiguous_velocity = (self.radar.wavelength / 2) * (prf / 2)
    range_axis = np.linspace(
        0, max_unambiguous_range, self.observation_height)
    velocity_axis = np.linspace(
        -max_unambiguous_velocity, max_unambiguous_velocity, self.observation_width)
    range_indices = np.abs(
        detections.range - range_axis[:, np.newaxis]).argmin(axis=0)
    velocity_indices = np.abs(
        detections.velocity - velocity_axis[:, np.newaxis]).argmin(axis=0)
    range_vel_map = np.zeros((len(range_axis), len(velocity_axis)))
    for i in range(len(detections)):
      range_vel_map[range_indices[i], velocity_indices[i]] += 1

    return range_vel_map
