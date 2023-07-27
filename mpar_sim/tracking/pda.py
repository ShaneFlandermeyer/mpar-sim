from typing import List, Tuple
import numpy as np
from mpar_sim.tracking.kalman import KalmanFilter
import numpy as np
from scipy.stats import multivariate_normal
from mpar_sim.tracking.gate import gate_volume, gate_threshold, ellipsoid_gate


class PDAFilter():
  def __init__(self,
               filter: KalmanFilter,
               pd: float = 0.90,
               pg: float = 0.99,
               clutter_density: float = None,
               ):
    self.filter = filter
    self.pd = pd
    self.pg = pg
    self.clutter_density = clutter_density

  def predict(self, dt: float, **kwargs):
    return self.filter.predict(dt=dt, **kwargs)

  def update(self,
             measurements: List[np.ndarray],
             predicted_state: np.ndarray,
             predicted_covar: np.ndarray,
             **state_filter_kwargs,
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
    _, _, innovation_covar, kalman_gain, predicted_measurement = \
      self.filter.update(
        measurement=np.empty(self.measurement_model.ndim),
        predicted_state=predicted_state,
        predicted_covar=predicted_covar,
        **state_filter_kwargs)

    # Gate measurements and compute clutter density (for non-parameteric PDA)
    gated_measurements = self.gate(measurements=measurements,
                                   predicted_measurement=predicted_measurement,
                                   innovation_covar=innovation_covar)
    if self.clutter_density:
      clutter_density = self.clutter_density
    else:
      m = len(gated_measurements)
      # For m validated measurements, the clutter density is m / V
      V_gate = gate_volume(innovation_covar=innovation_covar,
                           gate_probability=self.pg,
                           ndim=self.measurement_model.ndim)
      clutter_density = m / V_gate

    # Compute association probabilities for gated measurements
    if len(gated_measurements) == 0:
      probs = [1]
    else:
      probs = self._association_probs(
          z=gated_measurements,
          z_pred=predicted_measurement,
          S=innovation_covar,
          pd=self.pd,
          pg=self.pg,
          clutter_density=clutter_density,
      )

    state, covar = self._update_state(
        z=gated_measurements,
        x_pred=predicted_state,
        P_pred=predicted_covar,
        K=kalman_gain,
        z_pred=predicted_measurement,
        S=innovation_covar,
        probs=probs,
    )
    return state, covar

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
    v = np.dot(probs[1:], y)
    x_post = x_pred + K @ v

    # Bar-Shalom2009 - Equations 42-44
    S_mix = np.einsum('m, mi, mj->ij', probs[1:], y, y)
    Pc = P_pred - K @ S @ K.T
    Pt = K @ (S_mix - np.outer(v, v)) @ K.T
    P_post = probs[0]*P_pred + (1 - probs[0])*Pc + Pt

    return x_post, P_post

  def gate(self, 
           measurements: List[np.ndarray],
           predicted_measurement: np.ndarray,
           innovation_covar: np.ndarray) -> np.ndarray:
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
                             predicted_measurement=predicted_measurement,
                             innovation_covar=innovation_covar,
                             threshold=G)
    return [m for m, valid in zip(measurements, in_gate) if valid]

  @property
  def state(self):
    return self.filter.state

  @property
  def covar(self):
    return self.filter.covar

  @property
  def transition_model(self):
    return self.filter.transition_model

  @property
  def measurement_model(self):
    return self.filter.measurement_model
