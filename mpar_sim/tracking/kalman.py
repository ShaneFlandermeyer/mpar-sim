import datetime
from typing import Tuple, Union
import numpy as np

from mpar_sim.models.measurement.base import MeasurementModel
from mpar_sim.models.transition.base import TransitionModel


class KalmanFilter():
  def __init__(self,
               state: np.ndarray,
               covar: np.ndarray,
               transition_model: TransitionModel = None,
               measurement_model: MeasurementModel = None,
               ):
    self.state = state
    self.covar = covar
    self.transition_model = transition_model
    self.measurement_model = measurement_model

  def predict(self, dt: float) -> Tuple[np.ndarray]:

    F = self.transition_model.matrix(dt)
    Q = self.transition_model.covar(dt)

    self.predicted_state, self.predicted_covar = self._predict(
        x=self.state,
        P=self.covar,
        F=F,
        Q=Q,
    )
    return self.predicted_state, self.predicted_covar

  def update(self, measurement: np.ndarray):
    x_post, P_post, S, K, z_pred = self._update(
        x_pred=self.predicted_state,
        P_pred=self.predicted_covar,
        z=measurement,
        H=self.measurement_model.matrix(),
        R=self.measurement_model.covar(),
    )
    self.innovation_covar = S
    self.kalman_gain = K
    self.predicted_measurement = z_pred
    self.state = x_post
    self.covar = P_post

  @staticmethod
  def _predict(
      x: np.ndarray,
      P: np.ndarray,
      F: np.ndarray,
      Q: np.ndarray,
  ) -> Tuple[np.ndarray]:
    """
    Kalman predict step

    See: https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python/  blob/master/06-Multivariate-Kalman-Filters.ipynb

    Parameters
    ----------
    x : np.ndarray
        State vector
    P : np.ndarray
        Covariance
    F : np.ndarray
        State transition matrix
    Q : np.ndarray
        Transition model noise covariance
    Returns
    -------
    Tuple[np.ndarray]
        _description_
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
    Kalman filter update step

    See: 
    - https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python/blob/master/06-Multivariate-Kalman-Filters.ipynb
    - https://stonesoup.readthedocs.io/en/v0.1b5/stonesoup.updater.html?highlight=kalman#module-stonesoup.updater.kalman

    Parameters
    ----------
    x_pred : np.ndarray
        State prediction
    P_pred : np.ndarray
        Covariance prediction
    z : np.ndarray
        Measurement
    H : np.ndarray
        Measurement model matrix
    R : np.ndarray
        Measurement noise covariance
    Returns
    -------
    Tuple[np.ndarray]
        Updated state and covariance
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
