import datetime
from typing import Collection, Dict, List, Optional

import cv2
import gymnasium as gym
import numba as nb
import numpy as np
import pygame
from gymnasium import spaces

from mpar_sim.beam.common import beamwidth2aperture
from mpar_sim.common.coordinate_transform import sph2cart
from mpar_sim.looks.spoiled_look import SpoiledLook
from mpar_sim.models.transition.base import TransitionModel
from mpar_sim.particle.surveillance_pso import SurveillanceSwarm
from mpar_sim.radar import PhasedArrayRadar
from mpar_sim.types.detection import Clutter
from mpar_sim.types.groundtruth import GroundTruthPath, GroundTruthState


class ParticleSurveillance(gym.Env):
  metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

  def __init__(self,
               radar: PhasedArrayRadar,
               swarm: SurveillanceSwarm,
               # Target generation parameters
               transition_model: TransitionModel,
               preexisting_states: Collection[np.ndarray] = [],
               min_initial_n_targets: int = 50,
               max_initial_n_targets: int = 50,
               max_az_span: float = 20,
               max_el_span: float = 20,
               range_span: List[float] = [10e3, 20e3],
               velocity_span: List[float] = [-100, 100],
               birth_rate: float = 0,
               death_probability: float = 0,
               start_time: Optional[datetime.datetime] = None,
               # Swarm parameters
               mutation_rate: float = 0.01,
               mutation_alpha: float = 0.25,
               # Environment parameters
               randomize_initial_state: bool = False,
               n_confirm_detections: int = 3,
               # Gym-specific parameters
               n_obs_bins: int = 50,
               image_shape: int = (84, 84),
               seed: int = None,
               render_mode: str = None,
               ):

    self.radar = radar
    self.swarm = swarm
    # Target generation parameters
    self.transition_model = transition_model
    self.preexisting_states = preexisting_states
    self.min_initial_n_targets = min_initial_n_targets
    self.max_initial_n_targets = max_initial_n_targets
    self.max_az_span = max_az_span
    self.max_el_span = max_el_span
    self.range_span = range_span
    self.velocity_span = velocity_span
    self.birth_rate = birth_rate
    self.death_probability = death_probability
    self.start_time = start_time
    # Swarm parameters
    self.mutation_rate = mutation_rate
    self.mutation_alpha = mutation_alpha
    # Environment parameters
    self.randomize_initial_state = randomize_initial_state
    self.n_confirm_detections = n_confirm_detections
    self.n_obs_bins = n_obs_bins
    self.feature_size = 3*self.n_obs_bins
    self.image_shape = image_shape
    self.seed = seed

    # Pre-compute azimuth/elevation axis values needed to digitize the particles for the observation output
    self.az_axis = np.linspace(self.swarm.position_bounds[0][0],
                               self.swarm.position_bounds[1][0],
                               self.image_shape[0])
    self.el_axis = np.linspace(self.swarm.position_bounds[0][1],
                               self.swarm.position_bounds[1][1],
                               self.image_shape[1])

    self.observation_space = spaces.Box(
        low=0, high=1, shape=(self.feature_size,), dtype=np.float32)

    self.action_space = spaces.Box(
        low=np.array(
            [-1, -1, 0, 0, 0, 0, 0, 0]),
        high=np.array([1, 1, 1, 1, np.Inf, np.Inf, np.Inf, np.Inf]),
        shape=(8,),
        dtype=np.float32)
    assert render_mode is None or render_mode in self.metadata["render_modes"]
    self.render_mode = render_mode

    # PyGame objects
    self.window = None
    self.clock = None

  def step(self, action: np.ndarray):

    look = self._action_to_look(action)

    # Point the radar in the right direction
    self.radar.load_look(look)
    timestep = datetime.timedelta(seconds=look.dwell_time)

    # Add an RCS to each target
    # TODO: Each target should have an RCS update function
    for path in self.target_paths:
      path.states[-1].rcs = 10

    # Bias particles away from the current look direction
    self.swarm.dispersion_phase(
        steering_angles=np.array(
            [look.azimuth_steering_angle, look.elevation_steering_angle]),
        beamwidths=np.array(
            [look.azimuth_beamwidth, look.elevation_beamwidth]),
    )

    # Try to detect the remaining untracked targets
    detections = self.radar.measure(
        self.target_paths, noise=False, timestamp=self.time)

    reward = 0
    for detection in detections:
      # Update the detection count for this target. If a non-clutter target that has not been tracked (n_detections < n_confirm_detections), the agent receives a reward.
      if not isinstance(detection, Clutter):
        target_id = detection.groundtruth_path.id
        if target_id not in self.detection_count.keys():
          self.detection_count[target_id] = 0
        self.detection_count[target_id] += 1

        if self.detection_count[target_id] == self.n_confirm_detections:
          self.n_tracks_initiated += 1
          reward += 1

      if 2 <= self.detection_count[target_id] <= self.n_confirm_detections:
        az = detection.state_vector[0]
        el = detection.state_vector[1]
        rng = detection.state_vector[2]
        self.swarm.detection_phase(az, el, rng)

    # Apply a Gaussian mutation to a fraction of the swarm to improve exploration.
    if self.mutation_rate > 0:
      self.swarm.gaussian_mutation(alpha=self.mutation_alpha)

    # If multiple subarrays are scheduled to execute at once, the timestep will be zero. In this case, don't update the environment just yet.
    # For the single-beam case, this will always execute
    if timestep > datetime.timedelta(seconds=0):
      # Randomly drop targets
      for path in self.target_paths.copy():
        if self.np_random.uniform(0, 1) <= self.death_probability:
          self.target_paths.remove(path)
          if path.id in self.detection_count.keys():
            del self.detection_count[path.id]

      # Move targets forward in time
      # self._move_targets(timestep)

      # Randomly create new targets
      for _ in range(self.np_random.poisson(self.birth_rate)):
        target = self._new_target(self.time)
        self.target_paths.append(target)

      self.time += timestep

    # Update useful info
    self.target_history |= set(self.target_paths)
    self.detection_history.append(detections)

    # Create outputs
    observation = self._get_obs()
    info = self._get_info()

    # Terminate the episode when all targets have been detected at least n_detections_max times
    if len(self.detection_count) == len(self.target_paths) and \
            all(count >= self.n_confirm_detections for count in self.detection_count.values()):
      terminated = True
    else:
      terminated = False

    truncated = False

    if self.render_mode == "human":
      self._render_frame()

    return observation, reward, terminated, truncated, info

  def reset(self, seed=None, options=None):
    super().reset(seed=seed)

    self.time = self.start_time or datetime.datetime.now()
    self.radar.timestamp = self.time
    self.index = 0

    # Reset target cluster
    if self.randomize_initial_state:
      self.cluster_center = np.empty((2,))
      self.cluster_center[0] = self.np_random.uniform(
          self.radar.az_fov[0], self.radar.az_fov[1])
      self.cluster_center[1] = self.np_random.uniform(
          self.radar.el_fov[0], self.radar.el_fov[1])

      self.cluster_span = np.empty((2,))
      self.cluster_span[0] = self.np_random.uniform(0, self.max_az_span)
      self.cluster_span[1] = self.np_random.uniform(0, self.max_el_span)

    self._initialize_targets()

    self.swarm.reset()

    # Reset metrics/helpful debug info
    self.target_history = set()
    self.detection_history = []
    self.detection_count = dict()
    self.n_tracks_initiated = 0

    observation = self._get_obs()
    info = self._get_info()

    if self.render_mode == "human":
      self._render_frame()

    return observation, info

  def render(self):
    if self.render_mode == "rgb_array":
      return self._render_frame()

  def close(self):
    if self.window is not None:
      pygame.display.quit()
      pygame.quit()

  ############################################################################
  # Internal gym-specific methods
  ############################################################################
  def _action_to_look(self, action: np.ndarray):
    # Convert action array into radar dwell parameters
    
    # Squash the actions into the range [-1, 1] then scale by the max/min az and el values
    az_steering_angle = action[0] * \
        (self.radar.az_fov[1] - self.radar.az_fov[0]) / 2
    el_steering_angle = action[1] * \
        (self.radar.el_fov[1] - self.radar.el_fov[0]) / 2

    # Convert the azimuth and elevation beamwidth inputs from the range [0, 1]
    az_beamwidth = action[2] * (self.radar.max_az_beamwidth -
                                self.radar.min_az_beamwidth) + self.radar.min_az_beamwidth
    el_beamwidth = action[3] * (self.radar.max_el_beamwidth -
                                self.radar.min_el_beamwidth) + self.radar.min_el_beamwidth

    # Compute the number of elements used to form the Tx beam. Assuming the total Tx power is equal to the # of tx elements times the max element power
    # tx_beamwidths =
    tx_aperture_size = beamwidth2aperture(
        np.array([az_beamwidth, el_beamwidth]), self.radar.wavelength) / self.radar.wavelength
    n_tx_elements = np.prod(np.ceil(
        tx_aperture_size / self.radar.element_spacing).astype(int))
    tx_power = n_tx_elements * self.radar.element_tx_power

    look = SpoiledLook(
        azimuth_steering_angle=az_steering_angle,
        elevation_steering_angle=el_steering_angle,
        azimuth_beamwidth=az_beamwidth,
        elevation_beamwidth=el_beamwidth,
        bandwidth=action[4],
        pulsewidth=action[5],
        prf=action[6],
        n_pulses=action[7],
        tx_power=tx_power,
        start_time=self.time,
    )
    return look

  def _particle_histogram(self) -> np.ndarray:
    az_indices = np.digitize(
        self.swarm.position[:, 0], self.az_axis, right=True)
    el_indices = np.digitize(
        self.swarm.position[:, 1], self.el_axis, right=True)
    flat_inds = az_indices * self.image_shape[0] + el_indices
    bin_counts = np.histogram(flat_inds, bins=np.arange(0, np.prod(self.image_shape)+1))[
        0].reshape(self.image_shape)
    return bin_counts, az_indices, el_indices

  @staticmethod
  @nb.njit
  def _make_state_vector(n_bins: int, 
                         positions: np.ndarray, 
                         ranges: np.ndarray, 
                         az_indices: np.ndarray, 
                         el_indices: np.ndarray, 
                         sorted_best_inds: np.ndarray) -> np.ndarray:
    """
    Compute the state vector from the bins with the most particles
    Parameters
    ----------
    n_bins : int
        _description_
    positions : np.ndarray
        _description_
    ranges : np.ndarray
        _description_
    az_indices : np.ndarray
        _description_
    el_indices : np.ndarray
        _description_
    sorted_best_inds : np.ndarray
        _description_

    Returns
    -------
    _type_
        _description_
    """
    state = np.zeros((n_bins, 3))
    for i in range(n_bins):
      mask = np.logical_and(
          az_indices == sorted_best_inds[0][i], el_indices == sorted_best_inds[1][i])
      state[i, 0] = np.mean(positions[mask, 0])
      state[i, 1] = np.mean(positions[mask, 1])
      state[i, 2] = np.mean(ranges[mask])

    return state

  def _get_obs(self) -> np.ndarray:
    """
    Convert swarm positions to an observation image

    Returns
    -------
    np.ndarray
        Output image where each pixel has a value equal to the number of swarm particles in that pixel.
    """
    # Compute the azimuths and elevations of the best bins (normalized to the range [-1, 1])
    bin_counts, az_indices, el_indices = self._particle_histogram()
    best_inds = np.argpartition(
        bin_counts, -self.n_obs_bins, axis=None)[-self.n_obs_bins:]
    sorted_best_inds = best_inds[np.argsort(
        bin_counts.flatten()[best_inds])][::-1]
    sorted_best_inds = np.unravel_index(sorted_best_inds, self.image_shape)

    obs = self._make_state_vector(n_bins=self.n_obs_bins,
                                  positions=self.swarm.position,
                                  ranges=self.swarm.range,
                                  az_indices=az_indices,
                                  el_indices=el_indices,
                                  sorted_best_inds=sorted_best_inds)
    obs = obs / np.array([np.max(self.az_axis), np.max(self.el_axis), 10e3])
    return obs.ravel()

  def _get_info(self):
    """
    Returns helpful info about the current state of the environment, including:

    - initation ratio: the fraction of targets in the scenario that have been initiated (i.e. detected at least n_confirm_detections times)
    - swarm_positions: the current positions of all the particles in the swarm
    """
    n_initiated_targets = np.sum(
        [count >= self.n_confirm_detections for count in self.detection_count.values()])
    initiation_ratio = n_initiated_targets / len(self.target_paths)
    return {
        "initiation_ratio": initiation_ratio,
        "n_tracks_initiated": self.n_tracks_initiated,
    }

  def _render_frame(self) -> Optional[np.ndarray]:
    """
    Draw the current observation in a PyGame window if render_mode is 'human', or return the pixels as a numpy array if not.

    Returns
    -------
    Optional[np.ndarray]
        Grayscale pixel representation of the observation if render_mode is 'rgb_array', otherwise None.
    """
    if self.window is None and self.render_mode == "human":
      pygame.init()
      pygame.display.init()
      self.window = pygame.display.set_mode((256, 256))

    if self.clock is None and self.render_mode == "human":
      self.clock = pygame.time.Clock()

    # Draw canvas from pixels
    # The observation gets inverted here because I want black pixels on a white background.
    pixels, _, _ = self._particle_histogram()
    pixels = np.flip(pixels.squeeze(), axis=1)
    pixels = cv2.resize(pixels, (256, 256), interpolation=cv2.INTER_NEAREST)
    canvas = pygame.surfarray.make_surface(pixels)

    if self.render_mode == "human":
      # Copy canvas drawings to the window
      self.window.blit(canvas, canvas.get_rect())
      pygame.event.pump()
      pygame.display.update()

      # Ensure that human rendering occurs at the pre-defined framerate
      self.clock.tick(self.metadata["render_fps"])
    else:
      return pixels

  ############################################################################
  # Scenario simulation methods
  ############################################################################
  def _initialize_targets(self):
    """
    Create new targets based on the initial state and preexisting states specified in the environment's input arguments
    """
    n_initial_targets = self.np_random.integers(
        self.min_initial_n_targets, self.max_initial_n_targets, endpoint=True)
    if self.preexisting_states or n_initial_targets > 0:
      # Use preexisting_states to make some ground truth paths
      preexisting_paths = [self._new_target(
          self.time, state_vector=state) for state in self.preexisting_states]

      # Simulate more groundtruth paths for the number of initial targets
      n_initial_targets -= len(self.preexisting_states)
      if n_initial_targets > 0:
        initial_simulated_paths = [self._new_target(
            self.time) for _ in range(n_initial_targets)]
      self.target_paths = preexisting_paths + initial_simulated_paths
    else:
      self.target_paths = []

  def _new_target(self,
                  time: datetime.datetime,
                  state_vector: Optional[np.ndarray] = None) -> GroundTruthPath:
    """
    Create a new target from the given state vector

    Parameters
    ----------
    time : datetime.datetime
        Time of target creation
    state_vector : np.ndarray, optional
        Target state where the position components are given in az/el/range in degrees and the velocities are in m/s, by default None

    Returns
    -------
    GroundTruthPath
        A new ground truth path starting at the target's initial state
    """

    state_vector = np.empty((6,))
    state_vector[self.radar.velocity_mapping] = self.np_random.uniform(
        self.velocity_span[0], self.velocity_span[1], size=3)

    # Randomly generate the target position based on the random cluster parameters
    az_span = [self.cluster_center[0] - self.cluster_span[0],
               self.cluster_center[0] + self.cluster_span[0]]
    el_span = [self.cluster_center[1] - self.cluster_span[1],
               self.cluster_center[1] + self.cluster_span[1]]
    state_vector[self.radar.position_mapping] = self.np_random.uniform(
        [az_span[0], el_span[0], self.range_span[0]],
        [az_span[1], el_span[1], self.range_span[1]])

    state_vector[self.radar.position_mapping] = np.clip(
        state_vector[self.radar.position_mapping],
        [self.radar.az_fov[0], self.radar.el_fov[0], self.radar.min_range],
        [self.radar.az_fov[1], self.radar.el_fov[1], self.radar.max_range])
    # TODO: This creates a bimodal distribution from the input state vector to test how the agent handles multiple target sources.
    # if self.np_random.uniform(0, 1) < 0.5:
    #   state_vector[self.radar.position_mapping[0:2]] *= -1

    # Convert target position from spherical to cartesian
    x, y, z = sph2cart(
        *state_vector[self.radar.position_mapping], degrees=True)
    state_vector[self.radar.position_mapping] = np.array([x, y, z])
    state = GroundTruthState(
        state_vector=state_vector,
        timestamp=time,
        metadata={'index': self.index})

    target_path = GroundTruthPath(states=[state])
    self.index += 1
    return target_path

  def _move_targets(self, dt: datetime.timedelta):
    """
    Move targets forward in time, removing targets that have left the radar's FOV
    """
    # Combine targets into one StateVectors object to vectorize transition update
    # NOTE: Asssumes all targets have the same transition model
    if len(self.target_paths) == 0:
      return
    state_vectors = np.hstack(
        [path[-1].state_vector.reshape((-1, 1)) for path in self.target_paths])
    updated_state_vectors = self.transition_model.function(
        state_vectors, noise=True, time_interval=dt)

    for itarget in range(len(self.target_paths)):
      updated_state = GroundTruthState(
          state_vector=updated_state_vectors[:, itarget],
          timestamp=self.time,
          metadata=self.target_paths[itarget][-1].metadata)
      self.target_paths[itarget].append(updated_state)
