import copy
import datetime
from typing import Collection, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from ordered_set import OrderedSet
from stonesoup.base import Property
from stonesoup.models.transition.base import TransitionModel
from stonesoup.models.transition.linear import (
    CombinedLinearGaussianTransitionModel, ConstantVelocity, SingerApproximate)
from stonesoup.simulator.simple import SingleTargetGroundTruthSimulator
from stonesoup.types.array import CovarianceMatrix, StateVector
from stonesoup.types.groundtruth import GroundTruthPath, GroundTruthState
from stonesoup.types.numeric import Probability
from stonesoup.types.state import GaussianState
from mpar_sim.beam.beam import RectangularBeam
from mpar_sim.particle.global_best import IncrementalGlobalBestPSO
from mpar_sim.radar import PhasedArrayRadar
from mpar_sim.looks.look import Look
from pyswarms.base.base_single import SwarmOptimizer
import matplotlib.pyplot as plt


class RadarSurveillance(gym.Env):
  metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

  def __init__(self,
               radar: PhasedArrayRadar,
               transition_model: TransitionModel,
               initial_state: GaussianState,
               birth_rate: float = 1.0,
               death_probability: float = 0.01,
               preexisting_states: Collection[StateVector] = [],
               initial_number_targets: int = 0,
               swarm_optim=None,
               seed=None,
               render_mode=None):
    # Define environment-specific parameters
    self.radar = radar
    self.transition_model = transition_model
    self.initial_state = initial_state
    self.birth_rate = birth_rate
    self.death_probability = death_probability
    self.preexisting_states = preexisting_states
    self.initial_number_targets = initial_number_targets
    self.seed = seed

    # TODO: Let the user specify the image size
    self.observation_shape = (128, 128, 1)
    self.observation_space = spaces.Box(
        low=np.zeros(self.observation_shape, dtype=np.uint8),
        high=np.ones(self.observation_shape, dtype=np.uint8),
        dtype=np.uint8)

    # Currently, actions are limited to beam steering angles in azimuth and elevation
    # TODO: Make the action space include all look parameters
    self.action_space = spaces.Box(-90, 90, shape=(2,), dtype=np.float32)

    assert render_mode is None or render_mode in self.metadata["render_modes"]
    self.render_mode = render_mode

    if swarm_optim is None:
      self.swarm_optim = self._swarm_optim_default()

    # Pre-compute azimuth/elevation axis values needed to digitize the particles for the observation output
    self.az_axis = np.linspace(self.swarm_optim.bounds[0][0],
                               self.swarm_optim.bounds[1][0],
                               self.observation_shape[0])
    self.el_axis = np.linspace(self.swarm_optim.bounds[0][1],
                               self.swarm_optim.bounds[1][1],
                               self.observation_shape[1])

  def step(self, action):

    # Point the radar in the right direction
    self.look.azimuth_steering_angle = action[0]
    self.look.elevation_steering_angle = action[1]
    self.look.start_time = self.time
    self.radar.load_look(self.look)
    timestep = datetime.timedelta(seconds=self.look.dwell_time)

    # Add an RCS to each target
    # TODO: Each target should have an RCS update function
    for path in self.target_paths:
      path.states[-1].rcs = 10

    detections = self.radar.measure(self.target_paths, noise=True)
    
    # TODO: Periodically reset the personal best of each particle

    # Update the particle swarm output
    for det in detections:
      az = det.state_vector[1].degrees
      el = det.state_vector[0].degrees
      self.swarm_optim.optimize(
          self._distance_objective, detection_pos=np.array([az, el]))
      
    # Mutate particles based on Engelbrecht equations (16.66-16.67)
    sigma = 0.1*(self.swarm_optim.bounds[1][0] - self.swarm_optim.bounds[0][0])
    Pm = 0.05
    mutate = self.np_random.uniform(0, 1, size=self.swarm_optim.swarm.position.shape) < Pm
    self.swarm_optim.swarm.position[mutate] += self.np_random.normal(0, sigma, size=self.swarm_optim.swarm.position[mutate].shape)

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
    info = self._get_info()
    # TODO: Implement a real reward function
    reward = len(detections)
    terminated = False
    truncated = False
    return observation, reward, terminated, truncated, info

  def reset(self, seed=None, options=None):
    super().reset(seed=seed)

    self.time = self.initial_state.timestamp or datetime.datetime.now()
    self.radar.timestamp = self.time
    self.index = 0

    # Reset targets
    self._initialize_targets()

    # TODO: For now, all look parameters but the beam angles are fixed
    # TODO: This should pass through the resource manager to ensure that all parameters are consistent/can be allocated
    self.look = Look(
        # Beam parameters
        azimuth_steering_angle=0,
        elevation_steering_angle=0,
        azimuth_beamwidth=10,
        elevation_beamwidth=10,
        # Waveform parameters
        bandwidth=50e6,
        pulsewidth=10e-6,
        prf=5000,
        n_pulses=100,
        tx_power=100e5,
        # Scheduler parameters
        start_time=self.time,
        priority=0,
    )

    self.swarm_optim.reset()
    self.swarm_optim.swarm.pbest_cost = np.full(
        self.swarm_optim.swarm_size[0], np.inf)

    observation = self._get_obs()
    info = self._get_info()

    if self.render_mode == "human":
      self._render_frame()

    # Reset metrics/helpful debug info
    self.target_history = []
    self.detection_history = []

    return observation, info

  ############################################################################
  # Internal methods
  ############################################################################
  def _get_obs(self):
    az_indices = np.digitize(
        self.swarm_optim.swarm.position[:, 0], self.az_axis) - 1
    el_indices = np.digitize(
        self.swarm_optim.swarm.position[:, 1], self.el_axis) - 1
    obs = np.zeros(self.observation_shape, dtype=np.float32)
    obs[az_indices, el_indices] = 1
    return obs

  def _get_info(self):
    return {}

  def _render_frame():
    raise NotImplementedError

  def _distance_objective(self, swarm_pos, detection_pos):
    return np.linalg.norm(swarm_pos - detection_pos, axis=1)

  def _swarm_optim_default(self):
    options = {'c1': 0, 'c2': 0.5, 'w': 0.8}
    return IncrementalGlobalBestPSO(n_particles=500,
                                    dimensions=2,
                                    options=options,
                                    bounds=([-45, -45], [45, 45]),
                                    )

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

  def _new_target(self, time: datetime.datetime, state_vector: Optional[StateVector] = None) -> GroundTruthPath:
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
    _type_
        _description_
    """
    vector = state_vector or \
        self.initial_state.state_vector + \
        self.initial_state.covar @ \
        self.np_random.standard_normal(size=(self.initial_state.ndim, 1))

    target_path = GroundTruthPath()
    target_path.append(GroundTruthState(
        state_vector=vector,
        timestamp=time,
        metadata={"index": self.index})
    )
    # Increment target index
    self.index += 1
    return target_path

  def _move_targets(self, dt: datetime.timedelta):
    """
    Move targets forward in time
    """
    for path in self.target_paths:
      index = path[-1].metadata.get("index")
      updated_state = self.transition_model.function(
          path[-1], noise=True, time_interval=dt)
      path.append(GroundTruthState(
          updated_state, timestamp=self.time,
          metadata={"index": index}))


if __name__ == '__main__':
  # Target generation model
  transition_model = CombinedLinearGaussianTransitionModel([
      ConstantVelocity(10),
      ConstantVelocity(10),
      ConstantVelocity(0),
  ])
  initial_state_mean = StateVector([10e3, 10, 0, 0, 0, 0])
  initial_state_covariance = CovarianceMatrix(
      np.diag([200, 20, 200, 20, 2000, 10]))
  initial_state = GaussianState(
      initial_state_mean, initial_state_covariance)

  # Radar system object
  radar = PhasedArrayRadar(
      ndim_state=transition_model.ndim_state,
      position_mapping=(0, 2, 4),
      velocity_mapping=(1, 3, 5),
      position=np.array([[0], [0], [0]]),
      rotation_offset=np.array([[0], [0], [0]]),
      # Array parameters
      n_elements_x=32,
      n_elements_y=32,
      element_spacing=0.5,  # Wavelengths
      element_tx_power=100e3,
      # System parameters
      center_frequency=3e9,
      system_temperature=290,
      noise_figure=4,
      # Scan settings
      beam_shape=RectangularBeam,
      az_fov=[-60, 60],
      el_fov=[-60, 60],
      # Detection settings
      false_alarm_rate=1e-7,
      include_false_alarms=True
  )

  # Environment
  env = RadarSurveillance(
      radar=radar,
      transition_model=transition_model,
      initial_state=initial_state,
      birth_rate=0,
      death_probability=0,
      initial_number_targets=10)
  env.reset()

  # Agent
  from mpar_sim.agents.raster_scan import RasterScanAgent
  import numpy as np

  search_agent = RasterScanAgent(
      azimuth_scan_limits=np.array([-45, 45]),
      elevation_scan_limits=np.array([-45, 45]),
      azimuth_beam_spacing=1,
      elevation_beam_spacing=1,
      azimuth_beamwidth=10,
      elevation_beamwidth=10,
      bandwidth=100e6,
      pulsewidth=10e-6,
      prf=5e3,
      n_pulses=100,
  )

  for i in range(1000):
    look = search_agent.act(env.time)[0]
    az = look.azimuth_steering_angle
    el = look.elevation_steering_angle

    obs, reward, terminated, truncated, info = env.step(np.array([az, el]))

  # PLOTS

  # Plot the particle swarm history
  from pyswarms.utils.functions import single_obj as fx
  from pyswarms.utils.plotters import (
      plot_cost_history, plot_contour, plot_surface)
  from pyswarms.utils.plotters.formatters import Mesher
  from pyswarms.utils.plotters.formatters import Designer
  d = Designer(limits=[(-45, 45), (-45, 45)],
               label=['azimuth (deg.)', 'elevation (deg.)'])

  animation = plot_contour(pos_history=env.swarm_optim.pos_history,
                           designer=d,)
  # Plot the scenario
  from stonesoup.plotter import Plotter, Dimension

  # plt.figure()
  plotter = Plotter(Dimension.THREE)
  plotter.plot_sensors(radar, "Radar")
  plotter.plot_ground_truths(env.target_history, radar.position_mapping)
  plotter.plot_measurements(env.detection_history, radar.position_mapping)
  plt.show()
