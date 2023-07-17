from typing import Tuple

import numpy as np

from mpar_sim.common.sigma_points import merwe_scaled_sigma_points
from mpar_sim.models.measurement import MeasurementModel
from mpar_sim.models.transition import TransitionModel


class UnscentedKalmanFilter():
  def __init__(self,
               state: np.ndarray,
               covar: np.ndarray,
               transition_model: TransitionModel,
               measurement_model: MeasurementModel,
               ):
    self.state = state
    self.covar = covar
    self.transition_model = transition_model
    self.measurement_model = measurement_model

  def predict(self, dt: float):
    # Compute sigma points and weights
    self.sigmas, self.Wm, self.Wc = merwe_scaled_sigma_points(
        x=self.state,
        P=self.covar,
        alpha=0.5,
        beta=2,
        kappa=3-self.state.size,
    )
    self.sigmas_f = np.zeros_like(self.sigmas)
    for i in range(len(self.sigmas)):
      self.sigmas_f[i] = self.transition_model(self.sigmas[i], dt)

    self.predicted_state, self.predicted_covar = self.unscented_transform(
        sigmas=self.sigmas_f,
        Wm=self.Wm,
        Wc=self.Wc,
        noise_covar=self.transition_model.covar(dt))

  def update(self, measurement: np.ndarray):
    # Compute sigma points in measurement space
    n_sigma_points, ndim_state = self.sigmas_f.shape
    ndim_measurement = measurement.size
    self.sigmas_h = np.zeros((n_sigma_points, ndim_measurement))
    for i in range(n_sigma_points):
      self.sigmas_h[i] = self.measurement_model(self.sigmas_f[i])
    
    x_post, P_post, S, K, z_pred = self._update(
      x_pred=self.predicted_state,
      P_pred=self.predicted_covar,
      z=measurement,
      R=self.measurement_model.covar(),
      sigmas_f=self.sigmas_f,
      sigmas_h=self.sigmas_h,
      Wm=self.Wm,
      Wc=self.Wc,
    )
    self.innovation_covar = S
    self.kalman_gain = K
    self.predicted_measurement = z_pred
    self.state = x_post
    self.covar = P_post

  @staticmethod
  def unscented_transform(sigmas: np.ndarray,
                          Wm: np.ndarray,
                          Wc: np.ndarray,
                          noise_covar: np.ndarray
                          ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Use the unscented transform to compute the mean and covariance from a set of sigma points

    Parameters
    ----------
    sigmas : np.ndarray
        Array of sigma points, where each row is a point in N-d space.
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
    x = np.dot(Wm, sigmas)

    # Covariance computation
    y = sigmas - x[np.newaxis, :]
    P = np.einsum('k, ki, kj->ij', Wc, y, y)
    P += noise_covar

    return x, P

  @staticmethod
  def _update(
      x_pred: np.ndarray,
      P_pred: np.ndarray,
      z: np.ndarray,
      # Measurement parameters
      R: np.ndarray,
      # Sigma point parameters
      sigmas_f: np.ndarray,
      sigmas_h: np.ndarray,
      Wm: np.ndarray,
      Wc: np.ndarray,
  ) -> Tuple[np.ndarray]:
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
    sigmas_h : np.ndarray
        Sigma points in measurement space
    Wm : np.ndarray
        Mean weights from the prediction step
    Wc : np.ndarray
        Covariance weights from the prediction step
    measurement_model : callable
        Measurement function
    R : np.ndarray
        Measurement noise. For now, assumed to be a matrix

    Returns
    -------
    Tuple[np.ndarray]
        A tuple containing the following:
          - Updated state vector
          - Updated covariance matrix
          - Innovation covariance
          - Kalman gain
          - Predicted measurement
    """
    

    # Compute the mean and covariance of the measurement prediction using the unscented transform
    z_pred, S = UnscentedKalmanFilter.unscented_transform(
        sigmas=sigmas_h,
        Wm=Wm,
        Wc=Wc,
        noise_covar=R,
    )

    # Compute the cross-covariance of the state and measurements
    Pxz = np.einsum('k, ki, kj->ij',
                    Wc, sigmas_f - x_pred, sigmas_h - z_pred)

    # Update the state vector and covariance
    K = Pxz @ np.linalg.inv(S)
    x_post = x_pred + K @ (z - z_pred)
    P_post = P_pred - K @ S @ K.T
    return x_post, P_post, S, K, z_pred