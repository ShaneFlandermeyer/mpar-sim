from typing import Tuple
import numpy as np

from mpar_sim.models.measurement.base import MeasurementModel


def kalman_update(prior_state: np.ndarray,
                  prior_covar: np.ndarray,
                  measurement: np.ndarray,
                  measurement_model: MeasurementModel) -> Tuple[np.ndarray, np.ndarray]:
  """
  Perform the Kalman update step. See equations here:
  https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python/blob/master/06-Multivariate-Kalman-Filters.ipynb
  
  The posterior covariance is comput

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
