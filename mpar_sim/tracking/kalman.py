from typing import Tuple
import numpy as np

from mpar_sim.models.measurement.base import MeasurementModel


def kalman_predict(prior_state: np.ndarray,
                   prior_covar: np.ndarray,
                   transition_matrix: np.ndarray,
                   noise_covar: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
  """
  Kalman predict step.

  Equations: https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python/blob/master/06-Multivariate-Kalman-Filters.ipynb

  Parameters
  ----------
  prior_state : np.ndarray
      Prior state vector
  prior_covar : np.ndarray
      Prior covariance matrix
  transition_matrix : np.ndarray
      Transition model matrix
  noise_covar : np.ndarray
  Returns
  -------
  Tuple[np.ndarray, np.ndarray]
      - Predicted state vector
      - Predicted covariance matrix
  """

  if np.isscalar(transition_matrix):
    transition_matrix = np.array(transition_matrix)

  predicted_state = transition_matrix @ prior_state
  predicted_covar = transition_matrix @ prior_covar @ transition_matrix.T + noise_covar

  return predicted_state, predicted_covar


def kalman_update(prior_state: np.ndarray,
                  prior_covar: np.ndarray,
                  measurement: np.ndarray,
                  measurement_model: MeasurementModel) -> Tuple[np.ndarray, np.ndarray]:
  """
  Perform the Kalman update step. See equations here:
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
  measurement_model : MeasurementModel
      Model used to collect measurement

  Returns
  -------
  Tuple[np.ndarray, np.ndarray]
      - Predicted state vector
      - Predicted covariance
  """
  # Compute the residual
  prior_measurement = measurement_model.function(prior_state)
  residual = measurement - prior_measurement

  # Compute the Kalman gain
  measurement_matrix = measurement_model.matrix()
  measurement_covar = measurement_model.covar()
  measurement_cross_covar = prior_covar @ measurement_matrix.T
  innovation_covar = measurement_matrix @ measurement_cross_covar + measurement_covar
  kalman_gain = measurement_cross_covar @ np.linalg.inv(innovation_covar)

  # Compute the updated state and covariance
  posterior_mean = prior_state + kalman_gain @ residual
  posterior_covar = prior_covar - kalman_gain @ innovation_covar @ kalman_gain.T

  return posterior_mean, posterior_covar


def kalman_predict_update(prior_state: np.ndarray,
                          prior_covar: np.ndarray,
                          measurement: np.ndarray,
                          transition_matrix: np.ndarray,
                          noise_covar: np.ndarray,
                          measurement_model: MeasurementModel) -> Tuple[np.ndarray, np.ndarray]:
  """
  Perform the Kalman predict and update steps.

  Parameters
  ----------
  prior_state : np.ndarray
      Prior state vector
  prior_covar : np.ndarray
      Prior covariance matrix
  measurement : np.ndarray
      Actual measurement
  transition_matrix : np.ndarray
      Transition model matrix
  noise_covar : np.ndarray
  measurement_model : MeasurementModel
      Model used to collect measurement

  Returns
  -------
  Tuple[np.ndarray, np.ndarray]
      - Predicted state vector
      - Predicted covariance matrix
  """
  predicted_state, predicted_covar = kalman_predict(
      prior_state, prior_covar, transition_matrix, noise_covar)
  posterior_state, posterior_covar = kalman_update(
      predicted_state, predicted_covar, measurement, measurement_model)

  return posterior_state, posterior_covar
