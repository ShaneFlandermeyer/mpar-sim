import datetime
from typing import Tuple, Union
import numpy as np

from mpar_sim.models.measurement.base import MeasurementModel
from mpar_sim.models.transition.base import TransitionModel


def kalman_predict(x: np.ndarray,
                   P: np.ndarray,
                   transition_model: TransitionModel,
                   dt: float) -> Tuple[np.ndarray, np.ndarray]:
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

  F = transition_model.matrix(dt)
  Q = transition_model.covar(dt)

  predicted_state = F @ x
  predicted_covar = F @ P @ F.T + Q

  return predicted_state, predicted_covar


def kalman_update(x_pred: np.ndarray,
                  P_pred: np.ndarray,
                  z: np.ndarray,
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
  z_pred = measurement_model.function(x_pred)
  y = z - z_pred

  # Compute the Kalman gain and system uncertainty
  H = measurement_model.matrix()
  R = measurement_model.covar()
  S = H @ P_pred @ H.T + R
  K = P_pred @ H.T @ np.linalg.inv(S)

  # Compute the updated state and covariance
  x_post = x_pred + K @ y
  P_post = P_pred - K @ S @ K.T

  return x_post, P_post


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
