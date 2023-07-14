from typing import List, Tuple
import numpy as np
from mpar_sim.tracking.kalman import KalmanFilter
import numpy as np
from scipy.stats import multivariate_normal
from mpar_sim.tracking.gate import gate_volume, gate_threshold, ellipsoid_gate


class PDAFilter():
  def __init__(self,
               state_filter: KalmanFilter,
               pd: float = 0.90,
               pg: float = 0.99,
               clutter_density: float = None,
               ):
    self.filter = state_filter
    self.pd = pd
    self.pg = pg
    self.clutter_density = clutter_density
    
  def gate(self, measurements: List[np.ndarray]) -> np.ndarray:
    """
    Filter measurements that are not in the ellipsoidal gate region centered around the predicted measurement

    Parameters
    ----------
    measurements : List[np.ndarray]
        List of measurements

    Returns
    -------
    np.ndarray
        Filtered list of measurements
    """
    G = gate_threshold(pg=self.pg,
                       ndim=self.measurement_model.ndim)
    in_gate = ellipsoid_gate(measurements=measurements,
                             predicted_measurement=self.filter.z_pred,
                             innovation_covar=self.filter.S,
                             threshold=G)
    return [m for m, valid in zip(measurements, in_gate) if valid]

  def update(self,
             measurements: List[np.ndarray],
             dt: float,
             ) -> Tuple[np.ndarray]:
    """
    Use PDA to incorporate new measurements into the track state estimate

    Parameters
    ----------
    measurements : List[np.ndarray]
        New measurements from the current time step
    dt : float
        Time since last update

    Returns
    -------
    Tuple[np.ndarray]
        Posterior state vector and covariance
    """
    # Get the predicted state/covariance/measurement, along with the innovation covariance and Kalman gain. 
    # Since we aren't actually updating the filter posterior here, an empty value can be passed to the update method.
    self.filter.predict(dt)
    self.filter.update(measurement=np.empty(self.measurement_model.ndim))

    # Gate measurements and compute clutter density (for non-parameteric PDA)
    gated_measurements = self.gate(measurements)
    if self.clutter_density:
      clutter_density = self.clutter_density
    else:
      m = len(gated_measurements)
      # For m validated measurements, the clutter density is m / V
      V_gate = gate_volume(innovation_covar=self.filter.S,
                           gate_probability=self.pg,
                           ndim=self.measurement_model.ndim)
      clutter_density = m / V_gate

    # Compute association probabilities for gated measurements
    if len(gated_measurements) == 0:
      probs = [1]
    else:
      probs = self._association_probs(
          z=gated_measurements,
          z_pred=self.filter.z_pred,
          S=self.filter.S,
          pd=self.pd,
          pg=self.pg,
          clutter_density=clutter_density,
      )

    self.filter.x, self.filter.P = self._update_state(
        z=gated_measurements,
        x_pred=self.filter.x_pred,
        P_pred=self.filter.P_pred,
        K=self.filter.K,
        z_pred=self.filter.z_pred,
        S=self.filter.S,
        probs=probs,
    )
    return self.filter.x, self.filter.P

  @staticmethod
  def _association_probs(
      z: List[np.array],
      z_pred: np.array,
      S: np.array,
      pd: float,
      pg: float,
      clutter_density: float,
  ) -> np.ndarray:
    """
    Compute the association probabilities for each measurement in the list of gated measurements.

    Parameters
    ----------
    z : List[np.array]
        Gated measurements
    z_pred : np.array
        Predicted track measurement
    S : np.array
        Innovation covar
    pd : float
        Probability of detection
    pg : float
        Gate probability
    clutter_density : float
        Density of the spatial Poisson process that models the clutter

    Returns
    -------
    np.ndarray
        Length-m+1 array of association probabilities. The first element is the probability of no detection.
    """
    m = len(z)
    probs = np.empty(m+1)
    # Probability of no detection
    probs[0] = 1 - pd*pg
    # Probability of each detection from likelihood ratio
    l = multivariate_normal.pdf(
        z,
        mean=z_pred,
        cov=S,
    )
    l_ratio = l * pd / clutter_density
    probs[1:] = l_ratio

    # Normalize to sum to 1
    probs /= np.sum(probs)
    return probs

  @staticmethod
  def _update_state(
      z: List[np.array],
      # Filter parameters
      x_pred: np.array,
      P_pred: np.array,
      z_pred: np.array,
      K: np.array,
      S: np.array,
      probs: np.ndarray,
  ) -> Tuple[np.ndarray]:
    """
    Compute the posterior state and covariance for the given track as a Gaussian mixture

    Parameters
    ----------
    z : List[np.array]
      List of gated measurements
    x_pred : np.array
        Predicted state
    P_pred : np.array
        Predicted covar
    z_pred : np.array
        Predicted measurement
    K : np.array
        Kalman gain matrix
    S : np.array
        Innovation covar
    betas : np.ndarray
        Association probabilities

    Returns
    -------
    Tuple[np.ndarray]
        Posterior state and covar
    """
    # If there are no gated measurements, return the predicted state and covariance
    if len(z) == 0:
      return x_pred, P_pred

    # State estimation
    # Bar-Shalom2009 - Equations 39-40
    y = np.array(z) - z_pred
    v = np.einsum('m, mi->i', probs[1:], y)
    x_post = x_pred + K @ v

    # Bar-Shalom2009 - Equations 42-44
    betaz = np.einsum('m, mi->mi', probs[1:], y)
    S_mix = np.einsum('mi, mj->ij', betaz, y) - np.outer(v, v)
    Pc = P_pred - K @ S @ K.T
    Pt = K @ S_mix @ K.T
    P_post = probs[0]*P_pred + (1 - probs[0])*Pc + Pt

    return x_post, P_post
  
  def predict(self, dt: float):
    self.filter.predict(dt)

  @property
  def x(self):
    return self.filter.x

  @property
  def P(self):
    return self.filter.P

  @property
  def transition_model(self):
    return self.filter.transition_model

  @property
  def measurement_model(self):
    return self.filter.measurement_model