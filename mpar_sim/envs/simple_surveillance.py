import copy
import datetime
from typing import Collection, Dict, Optional
import cv2

import gymnasium as gym
import numpy as np
import pygame
from gymnasium import spaces
from stonesoup.types.state import GaussianState
from mpar_sim.beam.common import beamwidth2aperture

from mpar_sim.common.coordinate_transform import sph2cart
from mpar_sim.common.wrap_to_interval import wrap_to_interval
from mpar_sim.defaults import default_gbest_pso, default_lbest_pso
from mpar_sim.looks.look import Look
from mpar_sim.looks.spoiled_look import SpoiledLook
from mpar_sim.models.transition.base import TransitionModel
from mpar_sim.particle.swarm import ParticleSwarm
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
               #    initial_number_targets: int = 0,
               min_initial_n_targets: int = 50,
               max_initial_n_targets: int = 50,
               # Particle swarm parameters
               swarm: ParticleSwarm = None,
               beta_g: float = 0.05,
               w_disp_min: float = 0.25,
               w_disp_max: float = 0.95,
               c_disp: float = 1.5,
               w_det: float = 0.25,
               c_det: float = 1.0,
               mutation_rate: float = 0.01,
               mutation_alpha: float = 0.25,
               # Environment parameters
               randomize_initial_state: bool = False,
               max_random_az_covar: float = 10**2,
               max_random_el_covar: float = 10**2,
               n_confirm_detections: int = 3,
               # Gym-specific parameters
               n_obs_bins: int = 50,
               image_shape: int = (84, 84),
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
    swarm : SwarmOptimizer, optional
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
    # self.initial_number_targets = initial_number_targets
    self.min_initial_n_targets = min_initial_n_targets
    self.max_initial_n_targets = max_initial_n_targets
    self.n_confirm_detections = n_confirm_detections
    self.randomize_initial_state = randomize_initial_state
    self.max_random_az_covar = max_random_az_covar
    self.max_random_el_covar = max_random_el_covar
    self.n_obs_bins = n_obs_bins
    self.feature_size = self.n_obs_bins*3
    self.image_shape = image_shape
    self.seed = seed

    self.observation_space = spaces.Box(
        low=0, high=1, shape=(self.feature_size,), dtype=np.float32)

    self.action_space = spaces.Box(
        low=np.array(
            [-1, -1, 0, 0, 0, 0, 0, 0]),
        high=np.array([1, 1, np.Inf, np.Inf, np.Inf, np.Inf, np.Inf, np.Inf]),
        shape=(8,),
        dtype=np.float32)
    assert render_mode is None or render_mode in self.metadata["render_modes"]
    self.render_mode = render_mode

    if swarm is None:
      pos_bounds = np.array([[self.radar.az_fov[0], self.radar.el_fov[0]],
                             [self.radar.az_fov[1], self.radar.el_fov[1]]])
      # TODO: Enforce velocity bounds in the swarm update
      vel_bounds = [-1, 1]
      self.swarm = ParticleSwarm(n_particles=10_000,
                                 n_dimensions=2,
                                 position_bounds=pos_bounds,
                                 velocity_bounds=vel_bounds)
    else:
      self.swarm = swarm
    self.beta_g = beta_g
    self.w_disp_min = w_disp_min
    self.w_disp_max = w_disp_max
    self.c_disp = c_disp
    self.w_det = w_det
    self.c_det = c_det
    self.mutation_rate = mutation_rate
    self.mutation_alpha = mutation_alpha
    self.w_disps = np.ones_like(self.swarm.position)*w_disp_max

    # Pre-compute azimuth/elevation axis values needed to digitize the particles for the observation output
    self.az_axis = np.linspace(self.swarm.position_bounds[0][0],
                               self.swarm.position_bounds[1][0],
                               self.image_shape[0])
    self.el_axis = np.linspace(self.swarm.position_bounds[0][1],
                               self.swarm.position_bounds[1][1],
                               self.image_shape[1])

    # PyGame objects
    self.window = None
    self.clock = None

  def step(self, action: np.ndarray):
    # Squash the actions into the range [-1, 1] then scale by the max/min az and el values
    az_steering_angle = action[0] * \
        (self.radar.az_fov[1] - self.radar.az_fov[0]) / 2
    el_steering_angle = action[1] * \
        (self.radar.el_fov[1] - self.radar.el_fov[0]) / 2

    # Compute the number of elements used to form the Tx beam. Assuming the total Tx power is equal to the # of tx elements times the max element power
    tx_beamwidths = np.array([action[2], action[3]])
    tx_aperture_size = beamwidth2aperture(
        tx_beamwidths, self.radar.wavelength) / self.radar.wavelength
    n_tx_elements = np.prod(np.ceil(
        tx_aperture_size / self.radar.element_spacing).astype(int))
    tx_power = n_tx_elements * self.radar.element_tx_power

    look = SpoiledLook(
        azimuth_steering_angle=az_steering_angle,
        elevation_steering_angle=el_steering_angle,
        azimuth_beamwidth=action[2],
        elevation_beamwidth=action[3],
        bandwidth=action[4],
        pulsewidth=action[5],
        prf=action[6],
        n_pulses=action[7],
        tx_power=tx_power,
        start_time=self.time,
    )

    # Point the radar in the right direction
    self.radar.load_look(look)
    timestep = datetime.timedelta(seconds=look.dwell_time)

    # Add an RCS to each target
    # TODO: Each target should have an RCS update function
    for path in self.target_paths:
      path.states[-1].rcs = 10

    # Bias particles away from the current look direction
    steering_angles = np.array(
        [look.azimuth_steering_angle, look.elevation_steering_angle])
    min_az = look.azimuth_steering_angle - look.azimuth_beamwidth/2
    max_az = look.azimuth_steering_angle + look.azimuth_beamwidth/2
    min_el = look.elevation_steering_angle - look.elevation_beamwidth/2
    max_el = look.elevation_steering_angle + look.elevation_beamwidth/2
    in_beam = np.logical_and(
        np.logical_and(
            self.swarm.position[:, 0] >= min_az,
            self.swarm.position[:, 0] <= max_az),
        np.logical_and(
            self.swarm.position[:, 1] >= min_el,
            self.swarm.position[:, 1] <= max_el))
    # Set the velocity of each mutated particle radially away from the beam
    relative_pos = self.swarm.position[in_beam] - steering_angles
    velocity = relative_pos / np.linalg.norm(relative_pos, axis=1)[:, None]
    self.swarm.velocity[in_beam] = velocity * \
        self.np_random.uniform(
        0, [look.azimuth_beamwidth, look.elevation_beamwidth], size=velocity.shape)
    # Adaptively set the w_disp
    self.w_disps[in_beam] = np.clip(
        self.c_disp*self.w_disps[in_beam], 0, self.w_disp_max)

    self.swarm.update_position()
    self.swarm.velocity *= self.w_disps

    # Try to detect the remaining untracked targets
    detections = self.radar.measure(
        self.target_paths, noise=True, timestamp=self.time)

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

        # Update the swarm if an untracked target is detected.
        # The probability that a particle gets updated decays exponentially with its distance from the latest detection. If an update occurs, the particle moves radially towards the detection
        is_tracked = self.detection_count[detection.groundtruth_path.id] >= self.n_confirm_detections
      if isinstance(detection, Clutter) or not is_tracked:
        az = detection.state_vector[1]
        el = detection.state_vector[0]
        relative_pos = np.array([az, el]).reshape((1, -1)) - self.swarm.position
        distance = np.linalg.norm(relative_pos, axis=1)[:, None]
        move_probability = np.exp(-self.beta_g*distance).ravel()
        move_inds = self.np_random.uniform(
            0, 1, size=distance.size) < move_probability
        velocity = relative_pos / distance
        self.swarm.velocity[move_inds] = self.w_det*self.swarm.velocity[move_inds] + self.c_det*velocity[move_inds] * \
            self.np_random.uniform(
                0, 1, size=velocity[move_inds].shape)
        self.w_disps[move_inds] = self.w_disp_min
        self.swarm.update_position()

    # Apply a Gaussian mutation to a fraction of the swarm to improve exploration.
    if self.mutation_rate > 0:
      self._mutate_swarm(alpha=self.mutation_alpha)

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
      # terminated = False
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
  def bin_count_image(self) -> np.ndarray:
    az_indices = np.digitize(
        self.swarm.position[:, 0], self.az_axis, right=True)
    el_indices = np.digitize(
        self.swarm.position[:, 1], self.el_axis, right=True)
    flat_inds = az_indices * self.image_shape[0] + el_indices
    bin_counts = np.bincount(flat_inds, minlength=np.prod(
        self.image_shape)).reshape(self.image_shape)
    return bin_counts

  def _get_obs(self) -> np.ndarray:
    """
    Convert swarm positions to an observation image

    Returns
    -------
    np.ndarray
        Output image where each pixel has a value equal to the number of swarm particles in that pixel.
    """
    bin_counts = self.bin_count_image()
    best_inds = np.argsort(bin_counts, axis=None)[:-self.n_obs_bins-1:-1]
    best_inds = np.unravel_index(best_inds, self.image_shape)
    az = self.az_axis[best_inds[0]] / max(self.az_axis)
    el = self.el_axis[best_inds[1]] / max(self.az_axis)
    counts = bin_counts[best_inds] / np.max(bin_counts[best_inds])
    obs = np.array([az, el, counts]).T.ravel()
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
      self.window = pygame.display.set_mode((256, 256))

    if self.clock is None and self.render_mode == "human":
      self.clock = pygame.time.Clock()

    # Draw canvas from pixels
    # The observation gets inverted here because I want black pixels on a white background.
    pixels = self.bin_count_image()
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
    # state_vector = state_vector or self.np_random.multivariate_normal(
    #     mean=self.initial_state.state_vector.ravel(),
    #     cov=self.initial_state.covar)
    # Currently generating targets uniformly in the range given by the covariance matrix
    initial_state_range = np.sqrt(np.diag(np.array(self.initial_state.covar)))
    state_vector = self.np_random.uniform(-initial_state_range,
                                          initial_state_range) + self.initial_state.state_vector.ravel()
    state_vector = state_vector.ravel()
    state_vector[self.radar.position_mapping] = np.clip(
        state_vector[self.radar.position_mapping],
        [self.radar.az_fov[0], self.radar.el_fov[0], self.radar.min_range],
        [self.radar.az_fov[1], self.radar.el_fov[1], self.radar.max_range])
    # TODO: This creates a bimodal distribution from the input state vector to test how the agent handles multiple target sources.
    # if self.np_random.uniform(0, 1) < 0.5:
    #   state_vector[self.radar.position_mapping[0:2]] *= -1

    # Convert state vector from spherical to cartesian
    x, y, z = sph2cart(
        *state_vector[self.radar.position_mapping], degrees=True)
    state_vector[self.radar.position_mapping] = np.array([x, y, z])
    state = GroundTruthState(
        state_vector=state_vector,
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
        [path[-1].state_vector.reshape((-1, 1)) for path in self.target_paths])
    updated_state_vectors = self.transition_model.function(
        state_vectors, noise=True, time_interval=dt)

    for itarget in range(len(self.target_paths)):
      updated_state = GroundTruthState(
          state_vector=updated_state_vectors[:, itarget],
          timestamp=self.time,
          metadata=self.target_paths[itarget][-1].metadata)
      self.target_paths[itarget].append(updated_state)

  ############################################################################
  # Particle swarm methods
  ############################################################################

  # Mutate all particles that are in the current beam. Since the density of particles represents a sort of untracked target density, this represents the idea that we have searched this az/el region and unseen targets are unlikely to exist there. If lots of detections are found, many optimization steps will be carried out to bring particles back into the beam. Otherwise, the beam region should be mostly empty.
  def _mutate_swarm(self, alpha: float = 0.25):
    """
    Perform a Gaussian mutation on all particles that are in the current beam.

    Since the density of particles represents the density of untracked targets, this mutation means that areas that have recently been searched are not likely to contain unseen targets. If lots of detections are found after this step, particles will be drawn back to the beam region, which should encourage the agent to search there again. Otherwise, the agent will be encouraged to search other areas that have not been searched recently.

    Parameters
    ----------
    look : Look
        Contains the azimuth/elevation steering angles and beamwidths
    alpha : float, optional
        The mutation scaling factor, which dictates the fraction of the search space used in the standard deviation computation. By default 0.25
    """
    mutate_inds = self.np_random.uniform(
        0, 1, size=self.swarm.n_particles) < self.mutation_rate
    if mutate_inds.any():
      sigma = alpha*(self.swarm.position_bounds[1] -
                     self.swarm.position_bounds[0]).reshape(1, -1)
      sigma = np.repeat(sigma, np.count_nonzero(mutate_inds), axis=0)

      self.swarm.position[mutate_inds] += self.np_random.normal(
          np.zeros_like(sigma), sigma)

      # If the particle position exceeds the az/el bounds, wrap it back into the range on the other side. This improves diversity when detections are at the edge of the space compared to simple clipping.
      self.swarm.position[mutate_inds] = wrap_to_interval(
          self.swarm.position[mutate_inds],
          self.swarm.position_bounds[0], self.swarm.position_bounds[1])

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
