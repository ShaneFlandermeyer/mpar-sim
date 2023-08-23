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
from mpar_sim.tracking.jpda import JPDATracker

class SPADA_Williams():
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


class SPADA():
  """
  Single-sensor, single-step SPA-based PDA.
  """
  
  def __init__(self, beta: np.ndarray, zeta: np.ndarray):
    """
    Parameters
    ----------
    beta : np.ndarray
        nt x mt matrix of unnormalized PDA association probabilities. Each row corresponds to a target, and the first column represents false alarms.
    """
    assert beta.shape == zeta.shape
    self.beta = beta
    self.zeta = zeta
    self.nt = beta.shape[0]
    self.mt = beta.shape[1] - 1

    self.msg_m2t = np.ones((self.nt, self.mt))

  def step(self) -> None:
    """
    Single step of the SPADA algorithm, using equations (30) and (31) from Meyer2018.
    """
    b0 = self.beta[:, 0]
    z0 = self.zeta[:, 0]
    betas = self.beta[:, 1:]
    zetas = self.zeta[:, 1:]
    self.msg_t2m = betas / \
      (b0 + np.sum(betas * self.msg_m2t, axis=1) - betas * self.msg_m2t)
    self.msg_m2t = zetas / \
      (z0 + np.sum(zetas * self.msg_t2m, axis=0) - zetas * self.msg_t2m)
    
  @property
  def pa(self) -> np.ndarray:
    """
    Returns
    -------
    np.ndarray
      Target-oriented DA probabilities (Meyer2018, page 233)  
    """
    msg_m2t = np.concatenate((np.ones((self.nt, 1)), self.msg_m2t), axis=1)
    p = self.beta * msg_m2t / (np.sum(self.beta*msg_m2t, axis=1))[:, None]
    return p
  
  @property
  def pb(self) -> np.ndarray:
    """
    Returns
    -------
    np.ndarray
      Target-oriented DA probabilities (Meyer2018, page 233)  
    """
    msg_t2m = np.concatenate((np.ones((self.mt, 1)), self.msg_t2m), axis=1)
    p = msg_t2m / np.sum(msg_t2m, axis=1)[:, None]
    return p
  
  @staticmethod
  def g(pd: float, lz: np.ndarray) -> np.ndarray:
    return np.concatenate(([1 - pd], pd * lz))
  
  @staticmethod
  def h(lambda_fa: float, nt: int) -> np.ndarray:
    return np.concatenate(([lambda_fa], np.ones(nt)))
  

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

  spa_w = SPADA_Williams(nt, mt)

  # Compute likelihoods for each state/measurement pair
  lzs = []
  S = np.eye(2)
  for z_pred in zs:
    lzs.append(multivariate_normal.pdf(
        zs,
        mean=z_pred,
        cov=S,
    ))

  # TODO: The algorithm only works if I scale the betas by the false alarm density. Since this is a constant, it shouldn't matter
  g = np.empty((nt, mt+1))
  h = np.empty((mt, nt+1))
  lambda_fa = 0.1 / (spacing*nt)**2
  for i in range(nt):
    g[i] = SPADA.g(pd, lzs[i])
  for i in range(mt):
    h[i] = SPADA.h(lambda_fa, nt)
  spa_m = SPADA(beta=g, zeta=h)
    
  for _ in range(10):
    p = spa_w.step(psi=g[:, 1:]/lambda_fa)
    spa_m.step()
    pa = spa_m.pa
    pb = spa_m.pb
  # print(p.T - pa)
  print(pa)
 
  