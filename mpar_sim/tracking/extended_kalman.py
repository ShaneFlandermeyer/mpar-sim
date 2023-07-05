import datetime
from typing import Tuple, Union
import numpy as np

from mpar_sim.models.measurement.base import MeasurementModel
from mpar_sim.models.transition.base import TransitionModel


def extended_kalman_predict(x: np.ndarray,
                            P: np.ndarray,
                            transition_model: TransitionModel,
                            dt: float,
                            ) -> Tuple[np.ndarray, np.ndarray]:
  """
  Extended Kalman predict step.  

  Equations: https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python/blob/master/11-Extended-Kalman-Filters.ipynb  

  Parameters
  ----------
  state : np.ndarray
      Prior state vector
  covar : np.ndarray
      Prior covariance matrix
  transition_model : TransitionModel
      Transition model object. This object must define the following methods:
          - function(state, dt): Propagates the state vector forward by dt
          - jacobian(dt): Returns the jacobian of the transition function
          - covar(dt): Returns the transition noise covariance matrix
  time_interval : Union[float, datetime.timedelta]
      Time interval over which to predict  

  Returns
  -------
  Tuple[np.ndarray, np.ndarray]
       - Predicted state vector
       - Predicted covariance
  """
  F = transition_model.jacobian(dt)
  Q = transition_model.covar(dt)

  # Propagate the state forward in time
  x_pred = transition_model.function(x, dt)
  P_pred = F @ P @ F.T + Q

  return x_pred, P_pred


def extended_kalman_update(x_pred: np.ndarray,
                           P_pred: np.ndarray,
                           z: np.ndarray,
                           measurement_model: MeasurementModel,
                           ) -> Tuple[np.ndarray, np.ndarray]:
  """
  Perform the extended Kalman update step. Unlike the traditional Kalman filter, the extended Kalman filter works for nonlinear measurement models. 
  The EKF is almost identical to the KF, but the nonlinear measurement must be linearized by using the Jacobian for the measurement matrix H.

  See equations here:
  https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python/blob/master/11-Extended-Kalman-Filters.ipynb

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
      Model used to collect measurement. This object must define the following methods:
        - function(state): Convert the input from state space into measurement space 
        - jacobian(): Returns the jacobian of the measurement function
        - covar(): Returns the measurement noise covariance matrix

  Returns
  -------
  Tuple[np.ndarray, np.ndarray]
      - Predicted state vector
      - Predicted covariance
  """
  x_pred = x_pred.ravel()
  z = z.ravel()
  # Compute the residual
  z_pred = measurement_model(x_pred, noise=False)
  # TODO: This does not handle aliasing of angles properly
  y = z - z_pred.ravel()

  # Compute the Kalman gain
  JH = measurement_model.jacobian(x_pred)
  R = measurement_model.covar()
  S = JH @ P_pred @ JH.T + R
  K = P_pred @ JH.T @ np.linalg.inv(S)

  # Compute the updated state and covariance
  x_post = x_pred + K @ y
  P_post = P_pred - K @ S @ K.T

  return x_post, P_post
