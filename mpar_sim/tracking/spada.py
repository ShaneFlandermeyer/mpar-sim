# Implementation of SPA-based PDA (Meyer2018, Section 6)
# Assumptions:
#  - nt targets
#  - Single sensor
# Goal: Compute association probabilities p(a_k^(i) | z_1:k) as marginals of the joint posterior pdf f(x_1:k, a_1:k, b_1:k | z_1:k) using loopy SPA on the factor graph in Fig. 2(c).
# Message passing schedule:
#  - Each time step k considered individually

from mpar_sim.tracking.gate import gate_volume
from mpar_sim.tracking.kalman import KalmanFilter
from mpar_sim.types import Target
import numpy as np
import matplotlib.pyplot as plt
from mpar_sim.models.measurement import LinearMeasurementModel
from mpar_sim.types.track import Track
from scipy.stats import multivariate_normal

class SPADA():
  """
  Single-sensor, single-step SPA-based PDA.
  
  TODO: Extend to the multi-step case (see Williams2010, section 3.3)
  """
  def __init__(self, nt, mt, pg=1):
    self.mu_ba = np.ones((mt, nt))
    self.pg = pg

  def step(self, psi):
    m, n = self.mu_ba.shape
    prodfact = self.mu_ba * psi
    sumprod = 1 + np.sum(prodfact, axis=0)
    mu_ab = psi / (sumprod[None, :] - prodfact)
    sum_mu_ab = 1 + np.sum(mu_ab, axis=1)
    self.mu_ba = 1 / (sum_mu_ab[:, None] - mu_ab)
    p = np.concatenate((np.ones((1, n)), self.mu_ba * psi))
    return p / np.sum(p, axis=0)
    
  def psi_p(self, pd, lz, lambda_fa):
    return pd * lz / (lambda_fa + 1e-10)
    
  @staticmethod
  def psi_c(at, i, bt, j):
    return at == j and bt == i
  
  

if __name__ == '__main__':
  class SimpleTarget(Target):
    def __init__(self, pd, **kwargs):
      super().__init__(**kwargs)
      self.pd = pd

    def detection_probability(self) -> float:
      return self.pd
  # Set up target scenario
  spacing = 5
  nt = 6
  pd = 0.6
  targets = []
  xs, ys = np.meshgrid(np.arange(3)*spacing, np.arange(2)*spacing)
  xs, ys = xs.flatten(), ys.flatten()
  for i in range(nt):
    targets.append(SimpleTarget(pd, position=[xs[i], ys[i]]))

  # Generate measurements
  mm = LinearMeasurementModel(ndim_state=2, covar=np.eye(2))
  zs = mm([t.position for t in targets], noise=True)
  mt = len(zs)
  
  
  spa = SPADA(nt, mt)
  
  # Compute marginal association probabilities 
  ## Compute likelihoods for each state/measurement pair
  lzs = []
  S = np.eye(2)
  for z_pred in zs:
    lzs.append(multivariate_normal.pdf(
        zs,
        mean=z_pred,
        cov=S,
    ))
    
  psi = np.empty((nt, mt))
  for i in range(nt):
    lambda_fa = 0
    psi[i] = spa.psi_p(pd, lzs[i], lambda_fa)
    
  for _ in range(2):
    p = spa.step(psi=psi)
    print(p)