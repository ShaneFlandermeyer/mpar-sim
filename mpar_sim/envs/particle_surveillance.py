import copy
import datetime
from typing import Collection, Dict, Optional

import gymnasium as gym
import numpy as np
import pygame
from gymnasium import spaces
from ordered_set import OrderedSet
from pyswarms.base.base_single import SwarmOptimizer
from stonesoup.dataassociator.base import DataAssociator
from stonesoup.dataassociator.neighbour import (GNNWith2DAssignment,
                                                NearestNeighbour)
from stonesoup.deleter.base import Deleter
from stonesoup.hypothesiser.distance import DistanceHypothesiser
from stonesoup.initiator.base import Initiator
from stonesoup.measures import Euclidean, Mahalanobis
from stonesoup.models.transition.base import TransitionModel
from stonesoup.models.transition.linear import (
    CombinedLinearGaussianTransitionModel, ConstantVelocity, SingerApproximate)
from stonesoup.plotter import Dimension, Plotter
from stonesoup.predictor.kalman import KalmanPredictor
from stonesoup.types.array import CovarianceMatrix, StateVector, StateVectors
from stonesoup.types.detection import Clutter
from stonesoup.types.groundtruth import GroundTruthPath, GroundTruthState
from stonesoup.types.hypothesis import SingleHypothesis
from stonesoup.types.state import GaussianState, State
from stonesoup.types.track import Track
from stonesoup.types.update import GaussianStateUpdate, Update
from stonesoup.updater.base import Updater
from stonesoup.updater.kalman import ExtendedKalmanUpdater

from mpar_sim.common.coordinate_transform import sph2cart
from mpar_sim.defaults import default_gbest_pso, default_lbest_pso
from mpar_sim.looks.look import Look
from mpar_sim.looks.spoiled_look import SpoiledLook
from mpar_sim.radar import PhasedArrayRadar


class ParticleSurveillance(gym.Env):
  metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

  def __init__(self,
               radar: PhasedArrayRadar,
               # Tracking parameters
               initiator: Initiator,
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
    preexisting_states : Collection[StateVector], optional
        A list of deterministic target states that are generated every time the scenario is initialized. This can be useful if you want to simulate a specific set of target trajectories, by default []
    initial_number_targets : int, optional
        Number of targets generated at the start of the simulation, by default 0
    swarm_optim : SwarmOptimizer, optional
        Particle swarm optimizer object used to generate the state images, by default None
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
    self.randomize_initial_state = randomize_initial_state
    self.max_random_az_covar = max_random_az_covar
    self.max_random_el_covar = max_random_el_covar
    self.seed = seed

    # Tracking parameters
    self.initiator = initiator

    self.observation_space = spaces.Box(
        low=0, high=255, shape=(128, 128, 1), dtype=np.uint8)

    # Currently, actions are limited to beam steering angles in azimuth and elevation
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

    detections = self.radar.measure(
        self.target_paths, noise=True, timestamp=self.time)
    reward = 0
    if len(detections) > 0:
      # Filter out detections that have been associated with an existing track
      associations = self.initiator.data_associator.associate(
          self.tracks, detections, self.time)
      associated_detections = set()
      for track in self.tracks:
        association = associations[track]
        if association.measurement:
          post = self.initiator.updater.update(association)
          track.append(post)
          associated_detections.add(association.measurement)
        else:
          track.append(association.prediction)

      # Update tentative tracks with new detections
      new_tracks, particle_detections = self.initiator.initiate(
          detections - associated_detections, self.time)
      self.tracks |= new_tracks

      # Update the particle swarm to account for detections associated with tentative tracks
      for detection in particle_detections:
        az = detection.state_vector[1].degrees
        el = detection.state_vector[0].degrees
        self.swarm_optim.optimize(
            self._distance_objective, detection_pos=np.array([az, el]))

      reward += len(new_tracks)

    # Mutate particles based on Engelbrecht equations (16.66-16.67)
    Pm = 0.002
    mutate = self.np_random.uniform(
        0, 1, size=len(self.swarm_optim.swarm.position)) < Pm
    sigma = 0.25*(self.swarm_optim.bounds[1] -
                  self.swarm_optim.bounds[0])[np.newaxis, :]
    sigma = np.repeat(sigma, np.count_nonzero(mutate), axis=0)
    self.swarm_optim.swarm.position[mutate] += self.np_random.normal(
        np.zeros_like(sigma), sigma)

    # If multiple subarrays are scheduled to execute at once, the timestep will be zero. In this case, don't update the environment just yet.
    # For the single-beam case, this will always execute
    if timestep > datetime.timedelta(seconds=0):
      # Randomly drop targets
      deleted_targets = [path for path in self.target_paths if self.np_random.uniform(
          0, 1) <= self.death_probability]
      # Delete the targets from the detection count
      for path in deleted_targets:
        if path.id in self.detection_count.keys():
          del self.detection_count[path.id]
      self.target_paths.difference_update(deleted_targets)

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
    info = self._get_info()

    # Terminate the episode if tracks have been initiated for all targets
    if len(self.tracks) >= len(self.target_paths):
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
    self.tentative_tracks = set()
    self.tracks = set()

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
    obs = np.zeros(self.observation_space.shape, dtype=np.uint8)
    obs[az_indices, el_indices] = 255
    return obs

  def _get_info(self):
    """
    Returns helpful info about the current state of the environment, including:

    - initation ratio: the fraction of targets in the scenario that have been initiated (i.e. detected at least n_confirm_detections times)
    - swarm_positions: the current positions of all the particles in the swarm
    """
    return {}

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
    pixels = ~self._get_obs()
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
        Target state where the position components are given in az/el/range in degrees and the velocities are in m/s, by default None

    Returns
    -------
    GroundTruthPath
        A new ground truth path starting at the target's initial state
    """
    state = state_vector or \
        self.initial_state.state_vector + \
        np.sqrt(self.initial_state.covar) @ \
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
    # Combine targets into one StateVectors object to vectorize transition update
    # NOTE: Asssumes all targets have the same transition model
    states = [path[-1].state_vector for path in self.target_paths]
    if len(states) > 0:
      states = StateVectors(states)
    else:
      return
    new_states = self.transition_model.function(
        states, noise=True, time_interval=dt)

    for itarget, path in enumerate(self.target_paths):
      index = path[-1].metadata.get("index")
      path.append(GroundTruthState(
          new_states[:, itarget], timestamp=self.time,
          metadata={"index": index}))

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
