import datetime
from typing import Collection, Optional

import gymnasium as gym
import numpy as np
from attr import attrib, attrs
from attr.validators import instance_of
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
from mpar_sim.radar import PhasedArrayRadar
from mpar_sim.looks.look import Look


@attrs
class RadarSurveillance(gym.Env):
  metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

  # Target generation parameters
  radar = attrib(
      type=PhasedArrayRadar,
      validator=instance_of(PhasedArrayRadar)
  )
  transition_model = attrib(
      type=TransitionModel,
      validator=instance_of(TransitionModel)
  )
  initial_state = attrib(
      type=GaussianState,
      validator=instance_of(GaussianState)
  )
  birth_rate = attrib(
      type=float,
      default=1.0,
      validator=instance_of((int, float))
  )
  death_probability = attrib(
      type=Probability,
      default=0.1,
      validator=instance_of((Probability, float))
  )
  seed = attrib(type=int, default=None,)
  preexisting_states = attrib(
      type=Collection[StateVector],
      default=list(),
  )
  initial_number_targets = attrib(
      default=0,
      validator=instance_of(int)
  )

  def __init__(self, render_mode=None):
    # TODO: Define environment-specific parameters

    # TODO: Let the user specify the image size
    self.observation_shape = (128, 128, 1)
    self.observation_space = spaces.Box(
        low=np.zeros(self.observation_shape, dtype=np.float32),
        high=np.ones(self.observation_shape, dtype=np.float32),
        dtype=np.float32)

    # Currently, actions are limited to beam steering angles in azimuth and elevation
    self.action_space = spaces.Box(-90, 90, shape=(2,), dtype=np.float32)

    assert render_mode is None or render_mode in self.metadata["render_modes"]
    self.render_mode = render_mode

  def step(self, action):
    
    # Point the radar in the right direction
    self.look.azimuth_steering_angle = action[0]
    self.look.elevation_steering_angle = action[1]
    self.radar.load_look(self.look)
    timestep = datetime.timedelta(self.look.dwell_time)
    
    # Add an RCS to each target
    # TODO: Each target should have an RCS update function
    for path in self.target_paths:
      path.states[-1].rcs = 10
    
    detections = radar.measure(self.target_paths, noise=True)
    
    # TODO: Update the particle swarm
    
    # Randomly drop targets
    self.target_paths.difference_update(
        path for path in self.target_paths if self.np_random.rand() <= self.death_probability
    )

    # Move targets forward in time
    # TODO: timestep depends on the action taken!!!
    if timestep > datetime.timedelta(seconds=0):
      self._move_targets(timestep)

    # Randomly create new targets
    for _ in range(self.np_random.poisson(self.birth_rate)):
      target = self._new_target(self.time)
      self.target_paths.add(target)

    self.time += timestep

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
      prf=1500,
      n_pulses=10,
      tx_power=100e3,
      # Scheduler parameters
      start_time=self.time,
      priority=0,
    )

  ############################################################################
  # Internal methods
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
  initial_state_mean = StateVector([10e3, 10, 5e3, 0, 0, 0])
  initial_state_covariance = CovarianceMatrix(
      np.diag([200, 0, 200, 0, 2000, 50]))
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
      false_alarm_rate=1e-6,
      include_false_alarms=False
  )

  # Environment
  env = RadarSurveillance(
    radar=radar,
    transition_model=transition_model,
    initial_state=initial_state,)
  env.reset()
  env.step(np.array([0, 0]))
