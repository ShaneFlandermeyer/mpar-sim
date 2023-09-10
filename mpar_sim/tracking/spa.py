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

def spada(init_probs: np.ndarray, niter: int = 2) -> np.ndarray:
  """
  Use the sum-product algorithm to compute unnormalized joint association hypotheses between I known targets and M measurements.

  Parameters
  ----------
  init_probs : np.ndarray
    I x M array of unnormalized single-target association probabilities
  L : int, optional
      Number of message passing iterations to run, by default 2

  Returns
  -------
  np.ndarray
      I x (M+1) array of marginal association probabilities for each target/measurement pair.
  """
  psi = init_probs
  # Initialize message passing
  msg_i2m = psi[:, 1:] / \
      (psi[:, 0][:, np.newaxis] +
       np.sum(psi[:, 1:], axis=1, keepdims=True) - psi[:, 1:])

  for _ in range(niter):
    msg_m2i = 1 / (1 + np.sum(msg_i2m, axis=0) - msg_i2m)
    msg_i2m = psi[:, 1:] / \
        (psi[:, 0][:, np.newaxis] + np.sum(msg_m2i * psi[:, 1:],
         axis=0, keepdims=True) - msg_m2i * psi[:, 1:])

  marginal_probs = np.empty_like(init_probs)
  marginal_probs[:, 0] = 1
  marginal_probs[:, 1:] = msg_i2m

  return marginal_probs

class TotalSPA():
  # TODO: Implement me
  def likelihood():
    # TODO: Compute the likelihood for each target-measurement pair
    # a_i = 1, ..., Nm
    pass