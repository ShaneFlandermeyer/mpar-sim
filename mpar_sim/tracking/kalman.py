import datetime
from typing import Tuple, Union
import numpy as np

from mpar_sim.models.measurement.base import MeasurementModel
from mpar_sim.models.transition.base import TransitionModel


def kalman_predict(state: np.ndarray,
                   covar: np.ndarray,
                   transition_model: TransitionModel,
                   time_interval: Union[float, datetime.timedelta]) -> Tuple[np.ndarray, np.ndarray]:
  """
  Kalman predict step.

  Equations: https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python/blob/master/06-Multivariate-Kalman-Filters.ipynb

  Parameters
  ----------
  state : np.ndarray
      Prior state vector
  covar : np.ndarray
      Prior covariance matrix
  transition_model : TransitionModel
      Transition model object. This object must define the following methods:
        - matrix(dt): Returns the transition matrix
        - covar(dt): Returns the transition noise covariance matrix
  time_interval : Union[float, datetime.timedelta]
      Time interval over which to predict

  Returns
  -------
  Tuple[np.ndarray, np.ndarray]
      - Predicted state vector
      - Predicted covariance matrix
  """

  transition_matrix = transition_model.matrix(time_interval)
  noise_covar = transition_model.covar(time_interval)

  predicted_state = transition_matrix @ state
  predicted_covar = transition_matrix @ covar @ transition_matrix.T + noise_covar

  return predicted_state, predicted_covar


def kalman_update(state: np.ndarray,
                  covar: np.ndarray,
                  measurement: np.ndarray,
                  measurement_model: MeasurementModel) -> Tuple[np.ndarray, np.ndarray]:
  """
  Perform the Kalman update step. See equations here:
  https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python/blob/master/06-Multivariate-Kalman-Filters.ipynb

  The posterior covariance is computed using the equation from here:
  https://stonesoup.readthedocs.io/en/v0.1b5/stonesoup.updater.html?highlight=kalman#module-stonesoup.updater.kalman

  Parameters
  ----------
  state : np.ndarray
      Predicted state vector (mean)
  covar : np.ndarray
      Predicted covariance
  measurement : np.ndarray
      Actual measurement
  measurement_model : MeasurementModel
      Model used to collect measurement. This object must define the following methods:
        - function(state): Convert the input from state space into measurement space 
        - matrix(): Returns the measurement matrix
        - covar(): Returns the measurement noise covariance matrix

  Returns
  -------
  Tuple[np.ndarray, np.ndarray]
      - Predicted state vector
      - Predicted covariance
  """
  # Compute the residual
  prior_measurement = measurement_model.function(state)
  residual = measurement - prior_measurement

  # Compute the Kalman gain
  measurement_matrix = measurement_model.matrix()
  measurement_covar = measurement_model.covar()
  measurement_cross_covar = covar @ measurement_matrix.T
  innovation_covar = measurement_matrix @ measurement_cross_covar + measurement_covar
  kalman_gain = measurement_cross_covar @ np.linalg.inv(innovation_covar)

  # Compute the updated state and covariance
  posterior_mean = state + kalman_gain @ residual
  posterior_covar = covar - kalman_gain @ innovation_covar @ kalman_gain.T

  return posterior_mean, posterior_covar


def kalman_predict_update(state: np.ndarray,
                          covar: np.ndarray,
                          transition_model: TransitionModel,
                          time_interval: Union[float, datetime.timedelta],
                          measurement: np.ndarray,
                          measurement_model: MeasurementModel) -> Tuple[np.ndarray, np.ndarray]:
  """
  Perform the Kalman predict and update steps.

  Parameters
  ----------
  state : np.ndarray
      Prior state vector
  covar : np.ndarray
      Prior covariance matrix
  transition_model : TransitionModel
      Transition model object. This object must define the following methods:
        - matrix(dt): Returns the transition matrix
        - covar(dt): Returns the transition noise covariance matrix
  time_interval : Union[float, datetime.timedelta]
      Time interval over which to predict
  measurement : np.ndarray
      Actual measurement
  measurement_model : MeasurementModel
      Model used to collect measurement. This object must define the following methods:
        - function(state): Convert the input from state space into measurement space 
        - matrix(): Returns the measurement matrix
        - covar(): Returns the measurement noise covariance matrix

  Returns
  -------
  Tuple[np.ndarray, np.ndarray]
      - Predicted state vector
      - Predicted covariance matrix
  """
  predicted_state, predicted_covar = kalman_predict(
      state, covar, transition_model, time_interval)
  posterior_state, posterior_covar = kalman_update(
      predicted_state, predicted_covar, measurement, measurement_model)

  return posterior_state, posterior_covar
