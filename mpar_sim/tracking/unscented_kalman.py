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
               state_residual_fn: callable = np.subtract,
               measurement_residual_fn: callable = np.subtract,
               ):
    self.state = state
    self.covar = covar
    self.transition_model = transition_model
    self.measurement_model = measurement_model
    self.state_residual_fn = state_residual_fn
    self.measurement_residual_fn = measurement_residual_fn

  def predict(self, dt: float):
    # Compute sigma points and weights
    self.sigmas, self.Wm, self.Wc = merwe_scaled_sigma_points(
        x=self.state,
        P=self.covar,
        alpha=0.5,
        beta=2,
        kappa=3-self.state.size,
    )
    # Transform points to the prediction space
    self.sigmas_f = np.zeros_like(self.sigmas)
    for i in range(len(self.sigmas)):
      self.sigmas_f[i] = self.transition_model(self.sigmas[i], dt)

    self.predicted_state, self.predicted_covar = self.unscented_transform(
        sigmas=self.sigmas_f,
        Wm=self.Wm,
        Wc=self.Wc,
        noise_covar=self.transition_model.covar(dt),
        residual_fn=self.state_residual_fn,
    )

  def update(self, measurement: np.ndarray):
    # Transform sigma points to measurement space
    n_sigma_points, ndim_state = self.sigmas_f.shape
    ndim_measurement = measurement.size
    self.sigmas_h = np.zeros((n_sigma_points, ndim_measurement))
    for i in range(n_sigma_points):
      self.sigmas_h[i] = self.measurement_model(self.sigmas_f[i])

    # State update
    x_post, P_post, S, K, z_pred = self._update(
        x_pred=self.predicted_state,
        P_pred=self.predicted_covar,
        z=measurement,
        R=self.measurement_model.covar(),
        sigmas_f=self.sigmas_f,
        sigmas_h=self.sigmas_h,
        Wm=self.Wm,
        Wc=self.Wc,
        state_residual_fn=self.state_residual_fn,
        measurement_residual_fn=self.measurement_residual_fn,
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
                          noise_covar: np.ndarray,
                          residual_fn: callable = np.subtract,
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
    residual_fn : callable
        Function handle to compute the residual. This must be specified manually for nonlinear quantities such as angles, which cannot be subtracted directly. Default is np.subtract

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
    y = residual_fn(sigmas, x[np.newaxis, :])
    P = np.einsum('k, ki, kj->ij', Wc, y, y)
    P += noise_covar

    return x, P

  @staticmethod
  def _update(
      x_pred: np.ndarray,
      P_pred: np.ndarray,
      z: np.ndarray,
      R: np.ndarray,
      # Sigma point parameters
      sigmas_f: np.ndarray,
      sigmas_h: np.ndarray,
      Wm: np.ndarray,
      Wc: np.ndarray,
      state_residual_fn: callable,
      measurement_residual_fn: callable,
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
    z_mean, S = UnscentedKalmanFilter.unscented_transform(
        sigmas=sigmas_h,
        Wm=Wm,
        Wc=Wc,
        noise_covar=R,
        residual_fn=measurement_residual_fn,
    )

    # Compute the cross-covariance of the state and measurements
    Pxz = np.einsum('k, ki, kj->ij',
                    Wc, 
                    state_residual_fn(sigmas_f, x_pred), 
                    measurement_residual_fn(sigmas_h, z_mean))

    # Update the state vector and covariance
    y = measurement_residual_fn(z, z_mean)
    K = Pxz @ np.linalg.inv(S)
    x_post = x_pred + K @ y
    P_post = P_pred - K @ S @ K.T
    return x_post, P_post, S, K, z_mean
