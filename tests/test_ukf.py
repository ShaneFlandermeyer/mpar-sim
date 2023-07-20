import datetime
import numpy as np
from typing import Tuple, Union

import pytest
from mpar_sim.models.transition.constant_velocity import ConstantVelocity
from mpar_sim.types.track import Track
from mpar_sim.types.trajectory import Trajectory
from mpar_sim.types.state import State
import matplotlib.pyplot as plt
from mpar_sim.common.sigma_points import merwe_scaled_sigma_points
from mpar_sim.models.transition import TransitionModel
from mpar_sim.models.measurement import MeasurementModel
from mpar_sim.tracking.unscented_kalman import UnscentedKalmanFilter

class NonlinearMeasurementModel():
  def __init__(self, R):
    self.R = R
    
  def __call__(self, state, noise=False):
    if noise:
      noise = np.random.multivariate_normal(np.zeros(2), self.R)
    else:
      noise = 0
    x = state[0].item()
    y = state[2].item()
    azimuth = np.arctan2(y, x)
    range = np.sqrt(x**2 + y**2)
    return np.array([azimuth, range]) + noise
  
  def covar(self):
    return self.R

def test_ukf():
  # Generate ground truth
  n_steps = 20
  current_time = last_update = 0
  dt = 1
  seed = 0
  np.random.seed(seed)

  # Generate ground truth
  trajectory = Trajectory()
  trajectory.append(state=np.array([0, 1, 0, 1]),
                    covar=np.diag([1., 0.5, 1., 0.5]),
                    timestamp=current_time)
  transition_model = ConstantVelocity(ndim_pos=2,
                                      noise_diff_coeff=0.005,
                                      seed=seed)
  for i in range(n_steps):
    state = transition_model(trajectory.state, dt=dt, noise=True)
    timestamp = i*dt
    trajectory.append(state=state, timestamp=timestamp)

  # Generate measurements
  measurement_model = NonlinearMeasurementModel(
    R=np.diag([np.deg2rad(0.1), 0.1]))
  measurements = []
  for state in trajectory:
    z = measurement_model(state.state, noise=True)
    measurements.append(z)

  # Test the UKF
  current_time = last_update = 0
  ukf = UnscentedKalmanFilter(
    transition_model=transition_model,
    measurement_model=measurement_model)
  track_state = [trajectory[0].state]
  track_covar = [trajectory[0].covar]
  # track = Track()
  for i, m in enumerate(measurements):
    pred_state, pred_covar = ukf.predict(
      state=track_state[-1],
      covar=track_covar[-1],
      dt=current_time - last_update)
    state, covar = ukf.update(measurement=m,
                              predicted_state=pred_state,
                              predicted_covar=pred_covar)[:2]
    track_state.append(state)
    track_covar.append(covar)
    last_update = current_time
    current_time += dt
  
  true_states = np.stack([state.state for state in trajectory]).T
  track_states = np.stack([state for state in track_state])[1:].T
  track_pos = track_states[[0, 2]]
  track_vel = track_states[[1, 3]]
  pos_mse = np.mean(np.linalg.norm(true_states[[0, 2]] - track_pos, axis=1))
  vel_mse = np.mean(np.linalg.norm(true_states[[1, 3]] - track_vel, axis=1))
  
  assert pos_mse < 3
  assert vel_mse < 1
  
  
if __name__ == '__main__':
  pytest.main()