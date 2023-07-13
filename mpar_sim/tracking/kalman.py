import datetime
from typing import Tuple, Union
import numpy as np

from mpar_sim.models.measurement.base import MeasurementModel
from mpar_sim.models.transition.base import TransitionModel


class KalmanFilter():
  def __init__(self,
               x: np.ndarray,
               P: np.ndarray,
               transition_model: TransitionModel=None,
               measurement_model: MeasurementModel=None,
               ):
    self.x = x
    self.P = P
    self.transition_model = transition_model
    self.measurement_model = measurement_model

  def predict(self, dt: float) -> Tuple[np.ndarray]:

    F = self.transition_model.matrix(dt)
    Q = self.transition_model.covar(dt)

    self.x_pred, self.P_pred = self._predict(
        x=self.x,
        P=self.P,
        F=F,
        Q=Q,
    )

  def update(self, measurement: np.ndarray):
    x_post, P_post, S, K, z_pred = self._update(
        x_pred=self.x_pred,
        P_pred=self.P_pred,
        z=measurement,
        H=self.measurement_model.matrix(),
        R=self.measurement_model.covar(),
    )
    self.x = x_post
    self.P = P_post
    self.S = S
    self.K = K
    self.z_pred = z_pred

  @staticmethod
  def _predict(
      x: np.ndarray,
      P: np.ndarray,
      F: np.ndarray,
      Q: np.ndarray,
  ) -> Tuple[np.ndarray]:
    """
    Kalman predict step.
    Equations: https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python/  blob/master/06-Multivariate-Kalman-Filters.ipynb
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
    x_pred = F @ x
    P_pred = F @ P @ F.T + Q
    return x_pred, P_pred

  @staticmethod
  def _update(x_pred: np.ndarray,
              P_pred: np.ndarray,
              z: np.ndarray,
              H: np.ndarray,
              R: np.ndarray) -> Tuple[np.ndarray]:
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
    z_pred = H @ x_pred
    y = z - z_pred

    # Compute the Kalman gain and system uncertainty
    S = H @ P_pred @ H.T + R
    K = P_pred @ H.T @ np.linalg.inv(S)

    # Compute the updated state and covariance
    x_post = x_pred + K @ y
    P_post = P_pred - K @ S @ K.T

    return x_post, P_post, S, K, z_pred


# def kalman_predict(
#     x: np.ndarray,
#     P: np.ndarray,
#     F: np.ndarray,
#     Q: np.ndarray,
# ) -> Tuple[np.ndarray, np.ndarray]:
#   """
#   Kalman predict step.

#   Equations: https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python/blob/master/06-Multivariate-Kalman-Filters.ipynb

#   Parameters
#   ----------
#   state : np.ndarray
#       Prior state vector
#   covar : np.ndarray
#       Prior covariance matrix
#   transition_model : TransitionModel
#       Transition model object. This object must define the following methods:
#         - matrix(dt): Returns the transition matrix
#         - covar(dt): Returns the transition noise covariance matrix
#   time_interval : Union[float, datetime.timedelta]
#       Time interval over which to predict

#   Returns
#   -------
#   Tuple[np.ndarray, np.ndarray]
#       - Predicted state vector
#       - Predicted covariance matrix
#   """

#   x_pred = F @ x
#   P_pred = F @ P @ F.T + Q

#   return x_pred, P_pred


# def kalman_update(x_pred: np.ndarray,
#                   P_pred: np.ndarray,
#                   z: np.ndarray,
#                   H: np.ndarray,
#                   R: np.ndarray) -> Tuple[np.ndarray]:
#   """
#   Perform the Kalman update step. See equations here:
#   https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python/blob/master/06-Multivariate-Kalman-Filters.ipynb

#   The posterior covariance is computed using the equation from here:
#   https://stonesoup.readthedocs.io/en/v0.1b5/stonesoup.updater.html?highlight=kalman#module-stonesoup.updater.kalman

#   Parameters
#   ----------
#   state : np.ndarray
#       Predicted state vector (mean)
#   covar : np.ndarray
#       Predicted covariance
#   measurement : np.ndarray
#       Actual measurement
#   measurement_model : MeasurementModel
#       Model used to collect measurement. This object must define the following methods:
#         - function(state): Convert the input from state space into measurement space
#         - matrix(): Returns the measurement matrix
#         - covar(): Returns the measurement noise covariance matrix

#   Returns
#   -------
#   Tuple[np.ndarray, np.ndarray]
#       - Predicted state vector
#       - Predicted covariance
#   """
#   # Compute the residual
#   z_pred = H @ x_pred
#   y = z - z_pred

#   # Compute the Kalman gain and system uncertainty
#   S = H @ P_pred @ H.T + R
#   K = P_pred @ H.T @ np.linalg.inv(S)

#   # Compute the updated state and covariance
#   x_post = x_pred + K @ y
#   P_post = P_pred - K @ S @ K.T

#   return x_post, P_post, S, K, z_pred
