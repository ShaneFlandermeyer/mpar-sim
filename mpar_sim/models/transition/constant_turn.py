import functools
from typing import Union
import numpy as np
from scipy.linalg import block_diag


class ConstantTurn():
  def __init__(self,
               w: float,
               q: Union[float, np.ndarray],
               ndim_pos: int = 2,
               degrees: bool = True,
               seed: int = np.random.randint(0, 2**32-1),
               ):
    self.ndim_pos = ndim_pos
    self.ndim_state = self.ndim_pos*2
    self.q = q * np.ones(self.ndim_pos)
    self.w = w
    self.degrees = degrees
    # TODO: If the user specifies a z dimension in the state vector, just use a constant velocity model for that dimension
    self.ndim_state = 4
    self.np_random = np.random.RandomState(seed)

  def __call__(
      self,
      state: np.array,
      dt: float,
      noise: bool = False
  ) -> np.ndarray:
    next_state = np.dot(self.matrix(dt), state)
    if noise:
      next_state += self.sample_noise(dt).reshape(state.shape)
    return next_state
  
  @functools.lru_cache()
  def matrix(self, dt: float):
    w = np.deg2rad(self.w) if self.degrees else self.w
    CWT = np.cos(w*dt)
    SWT = np.sin(w*dt)
    SW = SWT / w
    CW = (1 - CWT) / w
    F = np.array([[1, SW, 0, CW],
                  [0, CWT, 0, -SWT],
                  [0, CW, 1, SW],
                  [0, SWT, 0, CWT]])
    return F

  @functools.lru_cache()
  def covar(self, dt: float):
    B = np.array([[0.25*dt**4, 0.5*dt**3],
                  [0.5**dt**3, dt**2]])
    Q = block_diag(*[B*q for q in self.q])
    return Q

  def sample_noise(self, dt: float) -> np.ndarray:
    noise = self.np_random.multivariate_normal(
        mean=np.zeros((self.ndim_state)), cov=self.covar(dt))
    return noise
