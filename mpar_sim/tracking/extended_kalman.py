from typing import Tuple, Callable
import numpy as np
from mpar_sim.models.measurement.linear import LinearMeasurementModel
from mpar_sim.models.measurement.nonlinear import NonlinearMeasurementModel


def extended_kalman_update(prior_state: np.ndarray,
                           prior_covar: np.ndarray,
                           measurement: np.ndarray,
                           measurement_noise_covar: np.ndarray,
                           measurement_function: Callable,
                           jacobian_function: Callable,
                           **kwargs,
                           ) -> Tuple[np.ndarray, np.ndarray]:
  """
  Perform the extended Kalman update step. Unlike the traditional Kalman filter, the extended Kalman filter works for nonlinear measurement models. 
  The EKF is almost identical to the KF, but the nonlinear measurement must be linearized by using the Jacobian for the measurement matrix H.

  See equations here:
  https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python/blob/master/06-Multivariate-Kalman-Filters.ipynb

  The posterior covariance is computed using the equation from here:
  https://stonesoup.readthedocs.io/en/v0.1b5/stonesoup.updater.html?highlight=kalman#module-stonesoup.updater.kalman

  Parameters
  ----------
  predicted_state : np.ndarray
      Predicted state vector (mean)
  predicted_covar : np.ndarray
      Predicted covariance
  measurement : np.ndarray
      Actual measurement
  measurement_noise_covar : np.ndarray
      Measurement model noise covariance matrix
  measurement_function : Callable
      Function used to go from state to measurement space
  jacobian_function : Callable
      Function used to compute the Jacobian of the measurement function

  Returns
  -------
  Tuple[np.ndarray, np.ndarray]
      - Predicted state vector
      - Predicted covariance
  """
  # Compute the residual
  prior_measurement = measurement_function(prior_state, **kwargs)
  residual = measurement - prior_measurement.ravel()

  # Compute the Kalman gain
  measurement_matrix = jacobian_function(prior_state, **kwargs)
  measurement_cross_covar = prior_covar @ measurement_matrix.T
  innovation_covar = measurement_matrix @ measurement_cross_covar + measurement_noise_covar
  kalman_gain = measurement_cross_covar @ np.linalg.inv(innovation_covar)

  # Compute the updated state and covariance
  posterior_mean = prior_state + kalman_gain @ residual
  posterior_covar = prior_covar - kalman_gain @ innovation_covar @ kalman_gain.T

  return posterior_mean, posterior_covar
