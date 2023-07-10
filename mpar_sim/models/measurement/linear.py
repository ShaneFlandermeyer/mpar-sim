import random
from typing import List

import jax
from mpar_sim.models.measurement import LinearMeasurementModel
import numpy as np

class LinearMeasurementModel(LinearMeasurementModel):
  def __init__(self,
               ndim_state: int,
               covar: np.array,
               measured_dims: List[int] = None,
               seed: int = None,
               ):
    self.ndim_state = ndim_state
    self.covar = covar
    if measured_dims is None:
      measured_dims = list(range(ndim_state))
    elif not isinstance(measured_dims, list):
      measured_dims = [measured_dims]
    self.measured_dims = measured_dims
    self.ndim = self.ndim_meas = len(measured_dims)
    
    if seed is None:
      seed = random.randint(0, 2**32-1)
    self.key = jax.random.PRNGKey(seed)
    
  def __call__(self, state: np.array, noise: bool = True):
    H = self.matrix()
    noise = self.sample_noise() if noise else 0
    return H @ state + noise
  
  def matrix(self):
    # NOTE: For now, using binary matrix with ones at measured inds
    H = np.zeros((self.ndim_meas, self.ndim_state))
    H[np.arange(self.ndim_meas), self.measured_dims] = 1
    return H
  
  def sample_noise(self):
    self.key, subkey = jax.random.split(self.key)
    noise = jax.random.multivariate_normal(
      key=subkey, mean=np.zeros(self.ndim), cov=self.covar)
    return noise
