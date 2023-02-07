import numpy as np
from mpar_sim.tracking.extended_kalman import extended_kalman_update
from mpar_sim.tracking.kalman import kalman_predict
from mpar_sim.models.transition.linear import ConstantVelocity
from mpar_sim.types.groundtruth import GroundTruthPath, GroundTruthState
from mpar_sim.models.measurement.nonlinear import CartesianToRangeAzElRangeRate
import matplotlib.pyplot as plt

if __name__ == '__main__':
  np.random.seed(1999)
  transition_model = ConstantVelocity(ndim_pos=3, noise_diff_coeff=0.05)
  measurement_model = CartesianToRangeAzElRangeRate(
      noise_covar=np.diag([0.1, 0.1, 1, 1]),
      discretize_measurements=False,
      alias_measurements=False)

  # Create the ground truth for testing
  truth = GroundTruthPath([GroundTruthState(np.array([50, 1, 0, 1, 0, 1]))])
  dt = 1.0
  for i in range(50):
    new_state = GroundTruthState(
        state_vector=transition_model.function(
            truth[-1].state_vector,
            noise=True,
            time_interval=dt)
    )
    truth.append(new_state)
  states = np.hstack([state.state_vector.reshape((-1, 1)) for state in truth])

  # Simulate measurements
  measurements = measurement_model.function(states, noise=True)

  # Test the tracking filter
  prior_state = np.array([50, 1, 0, 1, 0, 1])
  prior_covar = np.diag([1.5, 0.5, 1.5, 0.5, 1.5, 0.5])
  track = np.zeros((6, len(truth)))
  for i in range(len(truth)):
    # Predict step
    x_predicted, P_predicted = kalman_predict(prior_state=prior_state,
                                              prior_covar=prior_covar,
                                              transition_matrix=transition_model.matrix(
                                                  dt),
                                              noise_covar=transition_model.covar(dt))
    # Update step
    posterior_state, posterior_covar = extended_kalman_update(
        prior_state=x_predicted,
        prior_covar=P_predicted,
        measurement=measurements[:, i],
        measurement_model=measurement_model,
    )
    
    # Store the results and update the prior
    track[:, i] = posterior_state
    prior_state = posterior_state
    prior_covar = posterior_covar

  plt.figure()
  plt.plot(states[0], states[2], 'k-', label='Truth')
  plt.plot(track[0], track[2], 'bo-', label='Track')
  plt.xlabel('x')
  plt.ylabel('y')
  plt.legend()

  # plt.figure()
  # plt.plot(np.rad2deg(measurements[0]))
  # plt.xlabel('Time step')
  # plt.ylabel('Azimuth (degrees)')

  # plt.figure()
  # plt.plot(measurements[1])
  # plt.xlabel('Time step')
  # plt.ylabel('Range')
  
  plt.show()
