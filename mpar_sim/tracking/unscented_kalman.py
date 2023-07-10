import datetime
import numpy as np
from typing import Tuple, Union
from mpar_sim.models.transition.constant_velocity import ConstantVelocity
from mpar_sim.types.groundtruth import GroundTruthPath, GroundTruthState
import matplotlib.pyplot as plt


def merwe_scaled_sigma_points(mean: np.ndarray,
                              covar: np.ndarray,
                              # Merwe parameters
                              alpha: float,
                              beta: float,
                              kappa: float
                              ) -> Tuple[np.ndarray, ...]:
  """
  Compute sigma points (and the weights for each point) using Van der Merwe's algorithm

  Parameters
  ----------
  mean : np.ndarray
      Input mean
  covar : np.ndarray
      Input covariance
  alpha : float
      Van der Merwe alpha parameter
  beta : float
      Van der Merwe beta parameter
  kappa : float
      Van der Merwe kappa parameter

  Returns
  -------
  Tuple[np.ndarray, np.ndarray, np.ndarray]
      A tuple containing:
        - An array where each COLUMN contains the position of a sigma point. This is because the numpy cholesky function returns a lower triangular matrix by default.
        - An array containing the weights for each sigma point mean
        - An array containing the weights for each sigma point covariance
  """
  # Compute the sigma points
  n = mean.size
  lambda_ = alpha**2 * (n + kappa) - n
  U = np.linalg.cholesky((n + lambda_) * covar)

  mean = mean[:, np.newaxis] if mean.ndim == 1 else mean
  sigmas = np.concatenate([mean, mean - U, mean + U], axis=1)

  # Compute the weight for each point
  Wc = Wm = np.full(2*n+1, 1 / (2*(n+lambda_)))
  Wc[0] = lambda_ / (n + lambda_) + (1 - alpha**2 + beta)
  Wm[0] = lambda_ / (n + lambda_)

  return sigmas, Wm, Wc


def unscented_transform(sigmas: np.ndarray,
                        Wm: np.ndarray,
                        Wc: np.ndarray,
                        Q: np.ndarray
                        ) -> Tuple[np.ndarray, np.ndarray]:
  """
  Use the unscented transform to compute the mean and covariance from a set of sigma points

  Parameters
  ----------
  sigmas : np.ndarray
      Array of sigma points, where each COLUMN is a point.
  Wm : np.ndarray
      Mean weight matrix
  Wc : np.ndarray
      Covariance weight matrix
  Q : np.ndarray
      Process noise matrix

  Returns
  -------
  Tuple[np.ndarray, np.ndarray]
      A tuple containing:
        - Mean vector computed by applying the unscented transform to the input sigma points
        - The covariance computed by applying the unscented transform to the input sigma points
  """
  # Mean computation
  mean = np.dot(sigmas, Wm)

  # Covariance computation
  n, n_sigma_points = sigmas.shape
  covar = np.zeros((n, n))
  for i in range(n_sigma_points):
    y = sigmas[:, i] - mean
    covar += Wc[i] * np.outer(y, y)
  covar += Q

  return mean, covar


def ukf_predict(x: np.ndarray,
                P: np.ndarray,
                Q: np.ndarray,
                transition_func: callable,
                dt: float
                ) -> Tuple[np.ndarray, ...]:
  """
  Unscented Kalman filter prediction step

  Parameters
  ----------
  prior_state : np.ndarray
      State vector before prediction
  prior_covar : np.ndarray
      Covariance before prediction
  process_noise : np.ndarray
      Process noise. For now, assumed to be a matrix.
  transition_func : callable
      State transition function handle
  dt : float
      Time step for the prediction

  Returns
  -------
  Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
      A tuple containing the following:
        - Predicted state
        - Predicted covariance
        - Sigma points (projected forward in time)
        - Mean weights
        - Covariance weights
  """
  # Compute the sigma points and their weights
  n = x.size
  sigmas, Wm, Wc = merwe_scaled_sigma_points(
      x, P, alpha=0.5, beta=2, kappa=3-n)

  # Transform the sigma points using the process function
  sigmas_f = np.zeros_like(sigmas)
  for i in range(len(sigmas)):
    sigmas_f[:, i] = transition_func(sigmas[:, i], dt)

  x_pred, P_pred = unscented_transform(sigmas_f, Wm, Wc, Q)
  return x_pred, P_pred, sigmas_f, Wm, Wc


def ukf_update(
    x_pred: np.ndarray,
    P_pred: np.ndarray,
    z: np.ndarray,
    # Measurement parameters
    measurement_func: callable,
    R: np.ndarray,
    # Sigma point parameters
    sigmas_f: np.ndarray,
    Wm: np.ndarray,
    Wc: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
  """
  Unscented Kalman filter update step

  Parameters
  ----------
  measurement : np.ndarray
      New measurement to use for the update
  predicted_state : np.ndarray
      State vector after the prediction step
  predicted_covar : np.ndarray
      Covariance after the prediction step
  sigmas_f : np.ndarray
      Sigma points after the prediction step
  Wm : np.ndarray
      Mean weights from the prediction step
  Wc : np.ndarray
      Covariance weights from the prediction step
  measurement_func : callable
      Measurement function
  measurement_noise : np.ndarray
      Measurement noise. For now, assumed to be a matrix

  Returns
  -------
  Tuple[np.ndarray, np.ndarray]
      A tuple containing the following:
        - Updated state vector
        - Updated covariance matrix
  """
  ndim_state, n_sigma_points = sigmas_f.shape
  ndim_measurement = z.size

  # Transform sigma points into measurement space
  sigmas_h = np.zeros((ndim_measurement, n_sigma_points))
  for i in range(n_sigma_points):
    sigmas_h[:, i] = measurement_func(sigmas_f[:, i], R)

  # Compute the mean and covariance of the measurement prediction using the unscented transform
  predicted_measurement, predicted_measurement_covar = unscented_transform(
      sigmas_h, Wm, Wc, R)

  # Compute the cross-covariance of the state and measurements
  cross_covar = np.zeros((ndim_state, ndim_measurement))
  for i in range(n_sigma_points):
    cross_covar += Wc[i] * np.outer(sigmas_f[:, i] - x_pred,
                                    sigmas_h[:, i] - predicted_measurement)

  # Update the state vector and covariance
  kalman_gain = cross_covar @ np.linalg.inv(predicted_measurement_covar)
  updated_state = x_pred + \
      kalman_gain @ (z - predicted_measurement)
  updated_covar = P_pred - \
      kalman_gain @ predicted_measurement_covar @ kalman_gain.T
  return updated_state, updated_covar


def ukf_predict_update(x: np.ndarray,
                       P: np.ndarray,
                       z: np.ndarray,
                       # Process model
                       transition_func: callable,
                       Q: np.ndarray,
                       dt: float,
                       # Measurement model
                       measurement_func: callable,
                       R: np.ndarray,
                       ) -> Tuple[np.ndarray, np.ndarray]:
  x_pred, P_pred, sigmas_f, Wm, Wc = ukf_predict(
      x=x, P=P, Q=Q, transition_func=transition_func, dt=dt)

  x_post, P_post = ukf_update(
      x_pred=x_pred, P_pred=P_pred, z=z, measurement_func=measurement_func, R=R, sigmas_f=sigmas_f, Wm=Wm, Wc=Wc)

  return x_post, P_post


if __name__ == '__main__':
  np.random.seed(1999)
  transition_model = ConstantVelocity(ndim_pos=2, noise_diff_coeff=0.05)

  # Create the ground truth for testing
  truth = GroundTruthPath([GroundTruthState(np.array([50, 1, 0, 1]))])
  dt = 1.0
  for i in range(50):
    new_state = GroundTruthState(
        state_vector=transition_model(
            state=truth[-1].state_vector,
            noise=True,
            dt=dt)
    )
    truth.append(new_state)
  states = np.hstack([state.state_vector for state in truth])

  # Simulate measurements
  def measurement_func(state, noise_covar, noise=False):
    if noise:
      noise = np.random.multivariate_normal(np.zeros(2), noise_covar)
    else:
      noise = 0
    x = state[0].item()
    y = state[2].item()
    azimuth = np.arctan2(y, x)
    range = np.sqrt(x**2 + y**2)
    return np.array([azimuth, range]) + noise

  R = np.diag([np.deg2rad(0.001), 0.01])

  measurements = np.zeros((2, len(truth)))
  for i in range(len(truth)):
    measurements[:, i] = measurement_func(truth[i].state_vector, R, noise=True)

  # Test the UKF
  prior_state = np.array([50, 1, 0, 1])
  prior_covar = np.diag([1.5, 0.5, 1.5, 0.5])
  track = np.zeros((4, len(truth)))
  for i in range(measurements.shape[1]):
    measurement = measurements[:, i]
    post_state, post_covar = ukf_predict_update(
        x=prior_state,
        P=prior_covar,
        z=measurement,
        transition_func=transition_model,
        Q=transition_model.covar(dt),
        dt=dt,
        measurement_func=measurement_func,
        R=R)
    track[:, i] = post_state
    prior_state = post_state
    prior_covar = post_covar

  plt.figure()
  plt.plot(states[0], states[2], 'k-', label='Truth')
  plt.plot(track[0], track[2], 'bo-', label='Track')
  plt.xlabel('x')
  plt.ylabel('y')
  plt.legend()

  plt.figure()
  plt.plot(np.rad2deg(measurements[0]))
  plt.xlabel('Time step')
  plt.ylabel('Azimuth (degrees)')

  plt.figure()
  plt.plot(measurements[1])
  plt.xlabel('Time step')
  plt.ylabel('Range')
  plt.show()
