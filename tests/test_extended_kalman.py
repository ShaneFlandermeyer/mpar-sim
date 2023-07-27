import datetime
from typing import Union
import numpy as np
from mpar_sim.tracking.extended_kalman import extended_kalman_update
from mpar_sim.tracking.kalman import KalmanFilter
from mpar_sim.models.transition.constant_velocity import ConstantVelocity
from mpar_sim.types.trajectory import State
from mpar_sim.types.trajectory import Trajectory
from mpar_sim.models.measurement.nonlinear import CartesianToRangeVelocityAzEl
import pytest


def test_ekf_update():
  transition_model = ConstantVelocity(ndim_pos=3, q=0.05)
  measurement_model = CartesianToRangeVelocityAzEl(
      noise_covar=np.diag([0.1, 0.1, 0.1, 0.1]),
      discretize_measurements=False,
      alias_measurements=False)

  # Create the ground truth for testing
  # truth = Trajectory([GroundTruthState(np.array([50, 1, 0, 1, 0, 1]))])
  truth = Trajectory()
  truth.append(np.array([50, 1, 0, 1, 0, 1]))
  dt = 1.0
  for i in range(50):
    state = transition_model(truth[-1].state, dt=dt, noise=False)
    truth.append(state)
  states = np.array([state.state for state in truth])

  # Simulate measurements
  measurements = measurement_model(list(states), noise=False)
  measurements = np.array(measurements)

  # Test the tracking filter
  prior_state = np.array([50, 1, 0, 1, 0, 1])
  prior_covar = np.diag([1.5, 0.5, 1.5, 0.5, 1.5, 0.5])
  kf = KalmanFilter(
    transition_model=transition_model,
    measurement_model=None)
  track = np.zeros((len(truth), 6))
  track[0] = prior_state
  for i in range(1, len(truth)):
    # Predict step
    # kf.state, kf.covar = prior_state, prior_covar
    x_pred, P_pred = kf.predict(state=prior_state, covar=prior_covar, dt=dt)
    # Update step
    posterior_state, posterior_covar = extended_kalman_update(
        x_pred=x_pred,
        P_pred=P_pred,
        z=measurements[i],
        measurement_model=measurement_model
    )
    

    # Store the results and update the prior
    track[i] = posterior_state
    prior_state = posterior_state
    prior_covar = posterior_covar

  position_mapping = [0, 2, 4]
  velocity_mapping = [1, 3, 5]
  position_mse = np.mean(
      (track[:, position_mapping] - states[:, position_mapping])**2, axis=1)
  velocity_mse = np.mean(
      (track[:, velocity_mapping] - states[:, velocity_mapping])**2, axis=1)
  assert np.all(position_mse < 1e-6)
  assert np.all(velocity_mse < 1e-6)


if __name__ == '__main__':
  pytest.main()
