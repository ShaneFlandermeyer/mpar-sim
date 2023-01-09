import scipy
import numpy as np
from typing import Tuple


def merwe_scaled_sigma_points(mean: np.ndarray,
                              covar: np.ndarray,
                              # Merwe parameters
                              alpha: float,
                              beta: float,
                              kappa: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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

  sigmas = np.zeros((n, 2*n+1))
  sigmas[:, 0] = mean
  for i in range(n):
    sigmas[:, i+1] = mean - U[:, i]
    sigmas[:, n+i+1] = mean + U[:, i]

  # Compute the weight for each point
  Wc = Wm = np.full(2*n+1, 1 / (2*(n+lambda_)))
  Wc[0] = lambda_ / (n + lambda_) + (1 - alpha**2 + beta)
  Wm[0] = lambda_ / (n + lambda_)

  return sigmas, Wm, Wc


def unscented_transform(sigmas: np.ndarray,
                        Wm: np.ndarray,
                        Wc: np.ndarray,
                        process_noise: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
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
  covar += process_noise

  return mean, covar


def ukf_predict(prior_state: np.ndarray,
                prior_covar: np.ndarray,
                process_noise: np.ndarray,
                transition_func: callable,
                dt: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
  n = prior_state.size
  sigmas, Wm, Wc = merwe_scaled_sigma_points(
      prior_state, prior_covar, alpha=0.1, beta=2, kappa=3-n)

  # Transform the sigma points using the process function
  sigmas_f = np.zeros_like(sigmas)
  for i in range(len(sigmas)):
    sigmas_f[:, i] = transition_func(sigmas[:, i], dt)

  predicted_state, predicted_covar = unscented_transform(
      sigmas_f, Wm, Wc, process_noise)
  return predicted_state, predicted_covar, sigmas_f, Wm, Wc


def ukf_update(measurement: np.ndarray,
               predicted_state: np.ndarray,
               predicted_covar: np.ndarray,
               # Sigma point parameters
               sigmas_f: np.ndarray,
               Wm: np.ndarray,
               Wc: np.ndarray,
               # Measurement parameters
               measurement_func: callable,
               measurement_noise: np.ndarray,) -> Tuple[np.ndarray, np.ndarray]:
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
  ndim_measurement = measurement.size

  # Transform sigma points into measurement space
  sigmas_h = np.zeros((ndim_measurement, n_sigma_points))
  for i in range(n_sigma_points):
    sigmas_h[:, i] = measurement_func(sigmas_f[:, i])

  # Compute the mean and covariance of the measurement prediction using the unscented transform
  predicted_measurement, predicted_measurement_covar = unscented_transform(
      sigmas_h, Wm, Wc, measurement_noise)

  # Compute the cross-covariance of the state and measurements
  cross_covar = np.zeros((ndim_state, ndim_measurement))
  for i in range(n_sigma_points):
    cross_covar += Wc[i] * np.outer(sigmas_f[:, i] - predicted_state,
                                    sigmas_h[:, i] - predicted_measurement)

  # Update the state vector and covariance
  kalman_gain = cross_covar @ np.linalg.inv(predicted_measurement_covar)
  updated_state = predicted_state + \
      kalman_gain @ (measurement - predicted_measurement)
  updated_covar = predicted_covar - \
      kalman_gain @ predicted_measurement_covar @ kalman_gain.T
  return updated_state, updated_covar


def ukf_predict_update(prior_state: np.ndarray,
                       prior_covar: np.ndarray,
                       measurement: np.ndarray,
                       # Process model
                       transition_func: callable,
                       process_noise: np.ndarray,
                       dt: float,
                       # Measurement model
                       measurement_func: callable,
                       measurement_noise: np.ndarray,
                       ):
  predicted_state, predicted_covar, sigmas_f, Wm, Wc = ukf_predict(
      prior_state, prior_covar, process_noise, transition_func, dt)

  updated_state, updated_covar = ukf_update(
      measurement, predicted_state, predicted_covar, sigmas_f, Wm, Wc, measurement_func, measurement_noise)

  return updated_state, updated_covar


if __name__ == '__main__':
  # Simple constant-velocity transition model
  def transition_func(x, dt):
    F = np.array([[1, dt, 0, 0, 0, 0],
                  [0, 1, 0, 0, 0, 0],
                  [0, 0, 1, dt, 0, 0],
                  [0, 0, 0, 1, 0, 0],
                  [0, 0, 0, 0, 1, dt],
                  [0, 0, 0, 0, 0, 1]])
    return np.dot(F, x)

  def measurement_func(x):
    return x + 1

  x = np.zeros((6,))
  P = np.eye(6)
  dt = 0.1
  process_noise = measurement_noise = np.zeros_like(P)
  z = measurement_func(x)
  x, P = ukf_predict_update(x, P, z, transition_func,
                            process_noise, dt, measurement_func, measurement_noise)
