import copy
import datetime
from typing import Collection, Dict, Optional

import gymnasium as gym
import numpy as np
import pygame
from gymnasium import spaces
from pyswarms.base.base_single import SwarmOptimizer
from stonesoup.types.state import GaussianState

from mpar_sim.common.coordinate_transform import sph2cart
from mpar_sim.common.wrap_to_interval import wrap_to_interval
from mpar_sim.defaults import default_gbest_pso, default_lbest_pso
from mpar_sim.looks.spoiled_look import SpoiledLook
from mpar_sim.models.transition.base import TransitionModel
from mpar_sim.radar import PhasedArrayRadar
from mpar_sim.types.detection import Clutter
from mpar_sim.types.groundtruth import GroundTruthPath, GroundTruthState


class SimpleParticleSurveillance(gym.Env):
  metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

  def __init__(self,
               radar: PhasedArrayRadar,
               # Target generation parameters
               transition_model: TransitionModel,
               initial_state: GaussianState,
               birth_rate: float = 1.0,
               death_probability: float = 0.01,
               preexisting_states: Collection[np.ndarray] = [],
               initial_number_targets: int = 0,
               # Particle swarm parameters
               swarm_optim: SwarmOptimizer = None,
               # Environment parameters
               randomize_initial_state: bool = False,
               max_random_az_covar: float = 10,
               max_random_el_covar: float = 10,
               n_confirm_detections: int = 2,
               seed: int = None,
               render_mode: str = None,
               ):
    """
    An environment for simulating a radar surveillance scenario. Targets are generated with initial positions/velocities drawn from a Gaussian distribution and new targets are generated from a poisson process.

    The action space the environment includes all parameters needed to specify a look, and the observation space is a 256x256x1 grayscale image representing the particle swarm in az/el space. That is, each pixel is 255 if a particle is present at that location, and 0 otherwise.

    Parameters
    ----------
    radar : PhasedArrayRadar
        Radar used to simulate detections
    transition_model : TransitionModel
        Target state transition model
    initial_state : GaussianState
        Gaussian mean vector/covariance matrix that defines the initial state distribution for targets
    birth_rate : float, optional
        Lambda parameter of poisson target generation process that defines the rate of target generation per timestep, by default 1.0
    death_probability : float, optional
        Probability of death at each time step (per target), by default 0.01
    preexisting_states : Collection[np.ndarray], optional
        A list of deterministic target states that are generated every time the scenario is initialized. This can be useful if you want to simulate a specific set of target trajectories, by default []
    initial_number_targets : int, optional
        Number of targets generated at the start of the simulation, by default 0
    swarm_optim : SwarmOptimizer, optional
        Particle swarm optimizer object used to generate the state images, by default None
    n_confirm_detections: int, optional
        Number of detections required to confirm a target, by default 2.
        If every target in the current scenario has been confirmed this many times, the episode is terminated.
    randomize_initial_state: bool, optional
        If true, the initial state is randomized on every call to reset(), by default false
    max_random_az_covar: float, optional
        Specifies the maximum azimuth covariance when the initial state is randomized, by default 10
    max_random_el_covar: float, optional
        Specifies the maximum elevation covariance when the initial state is randomized, by default 10
    seed : int, optional
        Random seed used for the env's np_random member, by default None
    render_mode : str, optional
        If 'rgb_array', the observations are given as numpy arrays. If 'human', an additional PyGame window is created to show the observations in real time, by default None
    """
    # Define environment-specific parameters
    self.radar = radar
    self.transition_model = transition_model
    self.initial_state = initial_state
    self.birth_rate = birth_rate
    self.death_probability = death_probability
    self.preexisting_states = preexisting_states
    self.initial_number_targets = initial_number_targets
    self.n_confirm_detections = n_confirm_detections
    self.randomize_initial_state = randomize_initial_state
    self.max_random_az_covar = max_random_az_covar
    self.max_random_el_covar = max_random_el_covar
    self.seed = seed

    self.observation_space = spaces.Box(
        low=0, high=255, shape=(84, 84, 1), dtype=np.uint8)

    self.action_space = spaces.Box(
        low=np.array(
            [self.radar.az_fov[0], self.radar.el_fov[0], 0, 0, 0, 0, 0, 0]),
        high=np.array([self.radar.az_fov[1], self.radar.el_fov[1],
                      np.Inf, np.Inf, np.Inf, np.Inf, np.Inf, np.Inf]),
        shape=(8,),
        dtype=np.float32)
    assert render_mode is None or render_mode in self.metadata["render_modes"]
    self.render_mode = render_mode

    if swarm_optim is None:
      self.swarm_optim = default_lbest_pso()

    # Pre-compute azimuth/elevation axis values needed to digitize the particles for the observation output
    self.az_axis = np.linspace(self.swarm_optim.bounds[0][0],
                               self.swarm_optim.bounds[1][0],
                               self.observation_space.shape[0])
    self.el_axis = np.linspace(self.swarm_optim.bounds[0][1],
                               self.swarm_optim.bounds[1][1],
                               self.observation_space.shape[1])

    # PyGame objects
    self.window = None
    self.clock = None

  def step(self, action: np.ndarray):

    look = SpoiledLook(
        azimuth_steering_angle=action[0],
        elevation_steering_angle=action[1],
        azimuth_beamwidth=action[2],
        elevation_beamwidth=action[3],
        bandwidth=action[4],
        pulsewidth=action[5],
        prf=action[6],
        n_pulses=action[7],
        # TODO: For a spoiled look, this depends on the number of elements used to form the tx beam
        tx_power=self.radar.n_elements_x *
        self.radar.n_elements_y*self.radar.element_tx_power,
        start_time=self.time,
    )

    # Point the radar in the right direction
    self.radar.load_look(look)
    timestep = datetime.timedelta(seconds=look.dwell_time)

    # Add an RCS to each target
    # TODO: Each target should have an RCS update function
    for path in self.target_paths:
      path.states[-1].rcs = 10

    detections = self.radar.measure(self.target_paths, noise=True)

    reward = 0
    for detection in detections:
      # Update the detection count for this target
      # In practice, we won't know if the detection is from clutter or not. However, for the first simulation I'm assuming no false alarms so this is to make the reward calculation still work when the false alarm switch is on
      if not isinstance(detection, Clutter):
        target_id = detection.groundtruth_path.id
        if target_id not in self.detection_count.keys():
          self.detection_count[target_id] = 0
        self.detection_count[target_id] += 1

        if self.detection_count[target_id] == self.n_confirm_detections:
          self.n_tracks_initiated += 1
        
        # Reward the agent for detecting a target
        if 2 <= self.detection_count[target_id] <= self.n_confirm_detections:
          reward += 1 / len(self.target_paths)

      # Only update the swarm state for particles that have not been confirmed
      if isinstance(detection, Clutter) or self.detection_count[target_id] <= self.n_confirm_detections:
        az = detection.state_vector[1, 0]
        el = detection.state_vector[0, 0]
        self.swarm_optim.optimize(
            self._distance_objective, detection_pos=np.array([az, el]))

    # Mutate particles based on Engelbrecht equations (16.66-16.67)
    Pm = 0.0025
    mutate = self.np_random.uniform(
        0, 1, size=len(self.swarm_optim.swarm.position)) < Pm

    if mutate.any():
      sigma = 0.25*(self.swarm_optim.bounds[1] -
                    self.swarm_optim.bounds[0])[np.newaxis, :]
      sigma = np.repeat(sigma, np.count_nonzero(mutate), axis=0)

      self.swarm_optim.swarm.position[mutate] += self.np_random.normal(
          np.zeros_like(sigma), sigma)
      # Clip the mutated values to ensure they don't go outside the bounds
      self.swarm_optim.swarm.position[mutate] = np.clip(
          self.swarm_optim.swarm.position[mutate],
          self.swarm_optim.bounds[0], self.swarm_optim.bounds[1])

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
      self._move_targets(timestep)

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

    self.time = self.initial_state.timestamp or datetime.datetime.now()
    self.radar.timestamp = self.time
    self.index = 0

    # Reset targets
    if self.randomize_initial_state:
      az_idx, el_idx, range_idx = self.radar.position_mapping
      # Randomize the mean state vector
      self.initial_state.state_vector[az_idx] = self.np_random.uniform(
          self.radar.az_fov[0],
          self.radar.az_fov[1])
      self.initial_state.state_vector[el_idx] = self.np_random.uniform(
          self.radar.el_fov[0],
          self.radar.el_fov[1])
      # Randomize the covariance
      self.initial_state.covar[az_idx, az_idx] = self.np_random.uniform(
          0, self.max_random_az_covar)
      self.initial_state.covar[el_idx, el_idx] = self.np_random.uniform(
          0, self.max_random_el_covar)
    # print(self.initial_state.state_vector)
    self._initialize_targets()

    self.swarm_optim.reset()

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
  def _get_obs(self) -> np.ndarray:
    """
    Convert swarm positions to an observation image

    Returns
    -------
    np.ndarray
        Output image where each pixel has a value equal to the number of swarm particles in that pixel.
    """
    az_indices = np.digitize(
        self.swarm_optim.swarm.position[:, 0], self.az_axis, right=True)
    el_indices = np.digitize(
        self.swarm_optim.swarm.position[:, 1], self.el_axis, right=True)
    indices = az_indices * self.observation_space.shape[1] + el_indices
    obs = np.clip(np.bincount(indices, minlength=np.prod(
        self.observation_space.shape)).reshape(self.observation_space.shape),
        0, 255).astype(np.uint8)

    return obs

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
      self.window = pygame.display.set_mode(
          self.observation_space.shape[:2])

    if self.clock is None and self.render_mode == "human":
      self.clock = pygame.time.Clock()

    # Draw canvas from pixels
    # The observation gets inverted here because I want black pixels on a white background.
    pixels = self._get_obs()
    pixels = np.flip(pixels.squeeze(), axis=1)
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
    if self.preexisting_states or self.initial_number_targets:
      # Use preexisting_states to make some ground truth paths
      preexisting_paths = [self._new_target(
          self.time, state_vector=state) for state in self.preexisting_states]

      # Simulate more groundtruth paths for the number of initial targets
      initial_simulated_paths = [self._new_target(
          self.time) for _ in range(self.initial_number_targets)]
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
    state_vec = state_vector or \
        self.initial_state.state_vector.view(np.ndarray) + \
        np.sqrt(self.initial_state.covar.view(np.ndarray)) @ \
        self.np_random.standard_normal(size=(self.initial_state.ndim, 1))
    # Convert state vector from spherical to cartesian
    x, y, z = sph2cart(
        *state_vec[self.radar.position_mapping, :], degrees=True)
    state_vec[self.radar.position_mapping, :] = np.array([x, y, z])
    state = GroundTruthState(
        state_vector=state_vec,
        timestamp=time,
        metadata={'index': self.index})

    target_path = GroundTruthPath(states=[state])
    # Increment target index
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
        [path[-1].state_vector for path in self.target_paths])
    updated_state_vectors = self.transition_model.function(
        state_vectors, noise=True, time_interval=dt)

    for itarget in range(len(self.target_paths)):
      updated_state = GroundTruthState(
          updated_state_vectors[:, itarget][:, np.newaxis],
          timestamp=self.time,
          metadata=self.target_paths[itarget][-1].metadata)
      self.target_paths[itarget].append(updated_state)

  ############################################################################
  # Particle swarm methods
  ############################################################################

  def _distance_objective(self,
                          swarm_pos: np.ndarray,
                          detection_pos: np.ndarray) -> np.ndarray:
    """
    Compute the distance between each particle and the detection

    Parameters
    ----------
    swarm_pos : np.ndarray
        N x D array of positions for each of the N particles in a D-dimensional search space
    detection_pos : np.ndarray
        1 x D array of the D-dimensional position of the detection

    Returns
    -------
    np.ndarray
        The distance of each particle from the detection
    """
    return np.linalg.norm(swarm_pos - detection_pos, axis=1)
