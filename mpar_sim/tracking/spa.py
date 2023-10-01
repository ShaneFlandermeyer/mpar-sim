from typing import List, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pytest

from mpar_sim.models.measurement import LinearMeasurementModel
from mpar_sim.models.transition import ConstantVelocity
from mpar_sim.tracking import JPDATracker, KalmanFilter
from mpar_sim.types import FalseDetection, Track, Trajectory, TrueDetection
from mpar_sim.types.state import State
from scipy.stats import multivariate_normal


def spada(single_probs: np.ndarray, niter: int = 2) -> np.ndarray:
  """
  Use the sum-product algorithm to compute unnormalized joint association hypotheses between I known targets and M measurements.

  Parameters
  ----------
  single_probs : np.ndarray
    I x M array of unnormalized single-target association probabilities
  L : int, optional
      Number of message passing iterations to run, by default 2

  Returns
  -------
  np.ndarray
      I x (M+1) array of marginal association probabilities for each target/measurement pair.
  """
  
  phi = single_probs
  
  # Initialize message passing
  msg_i2m = phi[:, 1:] / \
      (phi[:, 0][:, np.newaxis] +
       np.sum(phi[:, 1:], axis=1, keepdims=True) - phi[:, 1:])
  for _ in range(niter):
    msg_m2i = 1 / (1 + np.sum(msg_i2m, axis=0) - msg_i2m)
    msg_i2m = phi[:, 1:] / (phi[:, 0][:, np.newaxis] + np.sum(msg_m2i *
                            phi[:, 1:], axis=1, keepdims=True) - msg_m2i * phi[:, 1:])

  marginal_probs = np.empty_like(single_probs)
  marginal_probs[:, 0] = 1
  marginal_probs[:, 1:] = msg_m2i

  return marginal_probs


class TotalSPA():
  def weights(self,
              pd: float,
              lz: np.ndarray,
              lam_c: float,
              niter: int = 2) -> np.ndarray:
    Nt, Nm = lz.shape
    pda_weights = self.likelihood(pd, lz, lam_c)
    # Uncomment for PDA (for comparison)
    # return pda_weights / np.sum(pda_weights, axis=1, keepdims=True)
    marginal_weights = spada(single_probs=pda_weights, niter=niter)
    w = np.empty((Nt, Nm+1))
    w[:, :-1] = pd * lz * marginal_weights[:, 1:] / lam_c
    w[:, -1] = (1 - pd) * marginal_weights[:, 0]
    w /= np.sum(w, axis=1, keepdims=True)
    return w


  def likelihood(self, pd: float, lz: np.ndarray, lam_c: float) -> np.ndarray:
    """
    Compute likelihood that a measurement comes from an object instead of clutter.

    These are 

    Parameters
    ----------
    pd : float
        Probability of detection, assumed to be the same for all targets
    lz : np.ndarray
        N x M array of likelihoods where index (n,m) indicates the likelihood that measurement m came from target n.
    lam_c : float
        Clutter rate, or expected number of clutter measurements per unit volume

    Returns
    -------
    np.ndarray
        N x (M+1) array of target-to-clutter likelihoods, where the first column is the probability that no measurement came from the target.
    """
    return np.append((1 - pd)*np.ones((lz.shape[0], 1)),
                     pd * lz / lam_c, axis=1)
