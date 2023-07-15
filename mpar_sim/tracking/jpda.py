from typing import List, Tuple
import numpy as np
from mpar_sim.tracking.kalman import KalmanFilter
import numpy as np
from scipy.stats import multivariate_normal
from mpar_sim.tracking.gate import gate_volume, gate_threshold, ellipsoid_gate
import functools
import itertools


class JPDAFilter():
  def __init__(self,
               filters: List[KalmanFilter],
               pd: float = 0.90,
               pg: float = 0.99,
               clutter_density: float = None,
               ):
    self.filters = filters
    self.pd = pd
    self.pg = pg
    self.clutter_density = clutter_density

  def gate(self,
           measurements: List[np.ndarray],
           predicted_measurement: np.ndarray,
           innovation_covar: np.ndarray,
           ) -> np.ndarray:
    """
    Filter measurements that are not in the ellipsoidal gate region centered around the predicted measurement

    Parameters
    ----------
    measurements : List[np.ndarray]
        List of measurements

    Returns
    -------
    np.ndarray
        A boolean array indicating whether each measurement is in the gate
    """
    G = gate_threshold(pg=self.pg,
                       ndim=predicted_measurement.size)
    in_gate = ellipsoid_gate(measurements=measurements,
                             predicted_measurement=predicted_measurement,
                             innovation_covar=innovation_covar,
                             threshold=G)
    return in_gate

  def update(self,
             measurements: List[np.ndarray],
             dt: float,
             ) -> Tuple[np.ndarray]:
    """
    Use PDA to incorporate new measurements into the track state estimate

    NOTE: For now, going to assume that the user has passed in one KF per track. Not sure how to handle adding more tracks

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
    measurement_inds = np.arange(1, len(measurements)+1)
    gated_measurements = []
    gated_inds = []
    single_probs = []
    for i, filt in enumerate(self.filters):
      # Get the predicted state/covariance/measurement, along with the innovation covariance and Kalman gain.
      # Since we aren't actually updating the filter posterior here, an empty value can be passed to the update method.
      filt.predict(dt=dt)
      filt.update(measurement=np.empty(filt.measurement_model.ndim))

      # Gate measurements and compute clutter density (for non-parameteric PDA)
      # TODO: Need to store a separate list of measurements for each track/gate
      in_gate = self.gate(
          measurements=measurements,
          predicted_measurement=filt.predicted_measurement,
          innovation_covar=filt.innovation_covar)
      gated_measurements.append([m for m, valid in zip(measurements, in_gate) if valid])
      if self.clutter_density:
        clutter_density = self.clutter_density
      else:
        m = len(gated_measurements[i])
        # For m validated measurements, the clutter density is m / V
        V_gate = gate_volume(innovation_covar=filt.innovation_covar,
                             gate_probability=self.pg,
                             ndim=filt.measurement_model.ndim)
        clutter_density = m / V_gate

      # Compute association probabilities for gated measurements
      if len(gated_measurements[i]) == 0:
        single_probs.append(np.ones(1))
      else:
        single_probs.append(self._association_probs(
            z=gated_measurements[i],
            z_pred=filt.predicted_measurement,
            S=filt.innovation_covar,
            pd=self.pd,
            pg=self.pg,
            clutter_density=clutter_density,
        ))
      gated_inds.append(np.append(0, measurement_inds[in_gate]))
      
    # Form single-track probabilities as a ntrack x nmeas+1 array
    probs = np.zeros((len(self.filters), len(measurements)+1))
    for i, single_prob in enumerate(single_probs):
      probs[i, gated_inds[i]] = single_prob
      
    # Compute joint probabilities for every valid combination of hypotheses
    hypotheses = list(itertools.product(*gated_inds))
    valid_hypotheses = np.array([h for h in hypotheses if self.isvalid(h)])
    joint_probs = np.prod(probs[np.arange(len(self.filters)), valid_hypotheses], axis=1)
    joint_probs /= np.sum(joint_probs)

    # Compute marginal probabilities for each track/measurement pair, then update the filter state and covar
    for i, filt in enumerate(self.filters):
      marginal_probs = np.bincount(valid_hypotheses[:, i], weights=joint_probs)
      filt.state, filt.covar = self._update_state(
            z=gated_measurements[i],
            x_pred=filt.predicted_state,
            P_pred=filt.predicted_covar,
            z_pred=filt.predicted_measurement,
            K=filt.kalman_gain,
            S=filt.innovation_covar,
            probs=marginal_probs[marginal_probs != 0],
        )
      
    return
      
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

  @staticmethod
  @functools.cache
  def isvalid(x: tuple) -> bool:
    """
    Checks if the input hypothesis tuple contains only unique values. Only the null hypothesis (0) can be repeated.

    Parameters
    ----------
    x : tuple
        _description_

    Returns
    -------
    bool
        _description_
    """
    x = np.array(x)
    x = x[x != 0]
    return len(set(x)) == len(x)
