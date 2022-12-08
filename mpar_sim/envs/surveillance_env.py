import copy
import datetime
from typing import Collection, Dict, Optional

import gymnasium as gym
import numpy as np
import pygame
from gymnasium import spaces
from ordered_set import OrderedSet
from stonesoup.models.transition.base import TransitionModel
from stonesoup.types.array import StateVector
from stonesoup.types.groundtruth import GroundTruthPath, GroundTruthState
from stonesoup.types.state import GaussianState
from stonesoup.types.detection import Clutter
from mpar_sim.common.coordinate_transform import sph2cart
from mpar_sim.defaults import default_gbest_pso, default_lbest_pso
from mpar_sim.looks.look import Look
from mpar_sim.radar import PhasedArrayRadar
from pyswarms.base.base_single import SwarmOptimizer


class ParticleSurveillance(gym.Env):
  metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

  def __init__(self,
               radar: PhasedArrayRadar,
               # Target generation parameters
               transition_model: TransitionModel,
               initial_state: GaussianState,
               birth_rate: float = 1.0,
               death_probability: float = 0.01,
               preexisting_states: Collection[StateVector] = [],
               initial_number_targets: int = 0,
               # Particle swarm parameters
               swarm_optim: SwarmOptimizer = None,
               # Environment parameters
               randomize_initial_state: bool = False,
               max_random_az_covar: float = 10,
               max_random_el_covar: float = 10,
               n_confirm_detections: int = 2,
               seed: int = None,
               render_mode: str = None):
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
    preexisting_states : Collection[StateVector], optional
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

    self.observation_shape = (256, 256, 1)
    self.observation_space = spaces.Box(
        low=np.zeros(self.observation_shape, dtype=np.uint8),
        high=255*np.ones(self.observation_shape, dtype=np.uint8),
        dtype=np.uint8)

    # Currently, actions are limited to beam steering angles in azimuth and elevation
    self.action_space = spaces.Dict(
        {
            "azimuth_steering_angle": spaces.Box(-90, 90, shape=(1,), dtype=np.float32),
            "elevation_steering_angle": spaces.Box(-90, 90, shape=(1,), dtype=np.float32),
            "azimuth_beamwidth": spaces.Box(0, np.inf, shape=(1,), dtype=np.float32),
            "elevation_beamwidth": spaces.Box(0, np.inf, shape=(1,), dtype=np.float32),
            "bandwidth": spaces.Box(0, np.inf, shape=(1,), dtype=np.float32),
            "pulsewidth": spaces.Box(0, np.inf, shape=(1,), dtype=np.float32),
            "prf": spaces.Box(0, np.inf, shape=(1,), dtype=np.float32),
            "n_pulses": spaces.Box(0, np.inf, shape=(1,), dtype=np.float32),
            "tx_power": spaces.Box(0, np.inf, shape=(1,), dtype=np.float32),
        }
    )

    assert render_mode is None or render_mode in self.metadata["render_modes"]
    self.render_mode = render_mode

    if swarm_optim is None:
      self.swarm_optim = default_lbest_pso()

    # Pre-compute azimuth/elevation axis values needed to digitize the particles for the observation output
    self.az_axis = np.linspace(self.swarm_optim.bounds[0][0],
                               self.swarm_optim.bounds[1][0],
                               self.observation_shape[0])
    self.el_axis = np.linspace(self.swarm_optim.bounds[0][1],
                               self.swarm_optim.bounds[1][1],
                               self.observation_shape[1])

    # PyGame objects
    self.window = None
    self.clock = None

  def step(self, action: Dict):

    look = self._action_dict_to_look(action)

    # Point the radar in the right direction
    self.radar.load_look(look)
    timestep = datetime.timedelta(seconds=look.dwell_time)

    # Add an RCS to each target
    # TODO: Each target should have an RCS update function
    for path in self.target_paths:
      path.states[-1].rcs = 10

    detections = self.radar.measure(self.target_paths, noise=True)

    cumulative_reward = 0
    for detection in detections:
      # Update the detection count for this target
      # TODO: In practice, we won't know if the detection is from clutter or not. However, for the first simulation I'm assuming no false alarms so this is to make the reward calculation still work when the false alarm switch is on
      if not isinstance(detection, Clutter):
        target_id = detection.groundtruth_path.id
        if target_id not in self.detection_count.keys():
          self.detection_count[target_id] = 0
        self.detection_count[target_id] += 1
        # Give a reward until the target has been detected too many times
        if self.detection_count[target_id] <= self.n_confirm_detections:
          cumulative_reward += 1

      # Update the particle swarm output
      az = detection.state_vector[1].degrees
      el = detection.state_vector[0].degrees
      self.swarm_optim.optimize(
          self._distance_objective, detection_pos=np.array([az, el]))

    # Mutate particles based on Engelbrecht equations (16.66-16.67)
    sigma = 0.1*(self.swarm_optim.bounds[1][0] - self.swarm_optim.bounds[0][0])
    Pm = 0.01
    mutate = self.np_random.uniform(
        0, 1, size=self.swarm_optim.swarm.position.shape) < Pm
    self.swarm_optim.swarm.position[mutate] += self.np_random.normal(
        0, sigma, size=self.swarm_optim.swarm.position[mutate].shape)

    # If multiple subarrays are scheduled to execute at once, the timestep will be zero. In this case, don't update the environment just yet.
    # For the single-beam case, this will always execute
    if timestep > datetime.timedelta(seconds=0):
      # Randomly drop targets
      self.target_paths.difference_update(
          path for path in self.target_paths if self.np_random.uniform(0, 1) <= self.death_probability
      )

      # Move targets forward in time
      self._move_targets(timestep)

      # Randomly create new targets
      for _ in range(self.np_random.poisson(self.birth_rate)):
        target = self._new_target(self.time)
        self.target_paths.add(target)

      self.time += timestep

    # Update useful info
    self.target_history |= self.target_paths
    self.detection_history.append(detections)

    # Create outputs
    observation = self._get_obs()
    reward = cumulative_reward
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
    self._initialize_targets()

    self.swarm_optim.reset()

    # Reset metrics/helpful debug info
    self.target_history = []
    self.detection_history = []
    self.detection_count = dict()

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
        Output image where each pixel is 1 if a particle is in that pixel and 0 otherwise.
    """
    az_indices = np.digitize(
        self.swarm_optim.swarm.position[:, 0], self.az_axis) - 1
    el_indices = np.digitize(
        self.swarm_optim.swarm.position[:, 1], self.el_axis) - 1
    obs = np.zeros(self.observation_shape, dtype=np.uint8)
    obs[az_indices, el_indices] = 255
    return obs

  def _get_info(self):
    return {}

  def _render_frame(self):
    if self.window is None and self.render_mode == "human":
      pygame.init()
      pygame.display.init()
      self.window = pygame.display.set_mode(
          self.observation_shape[:2])

    if self.clock is None and self.render_mode == "human":
      self.clock = pygame.time.Clock()

    # Draw canvas from pixels
    # The observation gets inverted here because I want black pixels on a white background.
    pixels = ~self._get_obs()
    canvas = pygame.surfarray.make_surface(pixels.squeeze())

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
      preexisting_paths = OrderedSet(
          self._new_target(self.time, state_vector=state) for state in self.preexisting_states)

      # Simulate more groundtruth paths for the number of initial targets
      initial_simulated_paths = OrderedSet(
          self._new_target(self.time) for _ in range(self.initial_number_targets))
      self.target_paths = preexisting_paths | initial_simulated_paths
    else:
      self.target_paths = OrderedSet()

  def _new_target(self,
                  time: datetime.datetime,
                  state_vector: Optional[StateVector] = None) -> GroundTruthPath:
    """
    Create a new target from the given state vector

    Parameters
    ----------
    time : datetime.datetime
        Time of target creation
    state_vector : StateVector, optional
        Target state, by default None

    Returns
    -------
    GroundTruthPath
        A new ground truth path starting at the target's initial state
    """
    state = state_vector or \
        self.initial_state.state_vector + \
        self.initial_state.covar @ \
        self.np_random.standard_normal(size=(self.initial_state.ndim, 1))
    # Convert state vector from spherical to cartesian
    x, y, z = sph2cart(*state[self.radar.position_mapping, :], degrees=True)
    state[self.radar.position_mapping, :] = np.array([x, y, z])[:, np.newaxis]

    target_path = GroundTruthPath()
    target_path.append(GroundTruthState(
        state_vector=state,
        timestamp=time,
        metadata={"index": self.index})
    )
    # Increment target index
    self.index += 1
    return target_path

  def _move_targets(self, dt: datetime.timedelta):
    """
    Move targets forward in time, removing targets that have left the radar's FOV
    """
    for path in self.target_paths:
      index = path[-1].metadata.get("index")
      updated_state = self.transition_model.function(
          path[-1], noise=True, time_interval=dt)
      path.append(GroundTruthState(
          updated_state, timestamp=self.time,
          metadata={"index": index}))

    # TODO: Remove target IDs from the dictionary if that target has left the FOV
    stale_targets = set()
    for path in self.target_paths:
      if not self.radar.is_detectable(path[-1]):
        stale_targets.add(path)
        if path.id in self.detection_count:
          del self.detection_count[path.id]

    self.target_paths.difference_update(stale_targets)

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

  def _action_dict_to_look(self, action: Dict) -> Look:
    """
    Convert an action dictionary with the necessary fields to a Look object

    Parameters
    ----------
    action : Dict
        Action dictionary

    Returns
    -------
    Look
        Look object that can be used to interface with the radar simulator
    """
    return Look(
        azimuth_steering_angle=action["azimuth_steering_angle"],
        elevation_steering_angle=action["elevation_steering_angle"],
        azimuth_beamwidth=action["azimuth_beamwidth"],
        elevation_beamwidth=action["elevation_beamwidth"],
        bandwidth=action["bandwidth"],
        pulsewidth=action["pulsewidth"],
        prf=action["prf"],
        n_pulses=action["n_pulses"],
        tx_power=action["tx_power"],
        start_time=self.time,
    )
