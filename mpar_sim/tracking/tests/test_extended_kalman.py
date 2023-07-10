import datetime
from typing import Union
import numpy as np
from mpar_sim.tracking.extended_kalman import extended_kalman_update
from mpar_sim.tracking.kalman import kalman_predict
from mpar_sim.models.transition.constant_velocity import ConstantVelocity
from mpar_sim.types.trajectory import State
from mpar_sim.types.trajectory import Trajectory
from mpar_sim.models.measurement.nonlinear import CartesianToRangeAzElVelocity
import pytest


def test_ekf_update():
  transition_model = ConstantVelocity(ndim_pos=3, noise_diff_coeff=0.05)
  measurement_model = CartesianToRangeAzElVelocity(
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
  states = np.stack([state.state for state in truth]).T

  # Simulate measurements
  measurements = measurement_model(states, noise=False)

  # Test the tracking filter
  prior_state = np.array([50, 1, 0, 1, 0, 1])
  prior_covar = np.diag([1.5, 0.5, 1.5, 0.5, 1.5, 0.5])
  track = np.zeros((6, len(truth)))
  track[:, 0] = prior_state
  for i in range(1, len(truth)):
    # Predict step
    x_predicted, P_predicted = kalman_predict(x=prior_state,
                                              P=prior_covar,
                                              transition_model=transition_model,
                                              dt=dt)
    # Update step
    posterior_state, posterior_covar = extended_kalman_update(
        x_pred=x_predicted,
        P_pred=P_predicted,
        z=measurements[:, i],
        measurement_model=measurement_model
    )

    # Store the results and update the prior
    track[:, i] = posterior_state
    prior_state = posterior_state
    prior_covar = posterior_covar

  position_mapping = [0, 2, 4]
  velocity_mapping = [1, 3, 5]
  position_mse = np.mean(
      (track[position_mapping] - states[position_mapping])**2, axis=1)
  velocity_mse = np.mean(
      (track[velocity_mapping] - states[velocity_mapping])**2, axis=1)
  assert np.all(position_mse < 1e-6)
  assert np.all(velocity_mse < 1e-6)


if __name__ == '__main__':
  pytest.main()
