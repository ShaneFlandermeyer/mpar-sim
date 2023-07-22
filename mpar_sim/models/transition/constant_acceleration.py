import numpy as np
from scipy.linalg import block_diag
import functools

class ConstantAcceleration():
  def __init__(self,
               ndim: int,
               q: float,
               seed: int = np.random.randint(0, 2**32-1),
               ):
    self.ndim = ndim
    self.ndim_state = 3*self.ndim
    self.q = q
    self.np_random = np.random.RandomState(seed)
    
  
  def __call__(
    self,
    state: np.ndarray,
    dt: float = 0,
    noise: bool = False,
  ) -> np.ndarray:
    next_state = np.dot(self.matrix(dt), state)
    if noise:
      next_state += self.sample_noise(dt).reshape(state.shape)
    return next_state
  
  @functools.lru_cache()
  def matrix(self, dt: float):
    F = np.array([[1, dt, 0.5*dt**2],
                  [0, 1, dt],
                  [0, 0, 1]])
    F = block_diag(*[F]*self.ndim)
    return F
  
  @functools.lru_cache()
  def covar(self, dt: float):
    Q = np.array([[dt**5/20, dt**4/8, dt**3/6],
                  [dt**4/8, dt**3/3, dt**2/2],
                  [dt**3/6, dt**2/2, dt]]) * self.q
    Q = block_diag(*[Q]*self.ndim)
    return Q
  
  def sample_noise(self, dt: float) -> np.ndarray:
    noise = self.np_random.multivariate_normal(
        mean=np.zeros((self.ndim_state)), cov=self.covar(dt))
    return noise
