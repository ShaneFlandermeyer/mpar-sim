from typing import List

import numpy as np

class LinearMeasurementModel():
  def __init__(self,
               ndim_state: int,
               covar: np.array,
               measured_dims: List[int] = None,
               seed: int = np.random.randint(0, 2**32-1),
               ):
    self.ndim_state = ndim_state
    self.noise_covar = covar
    if measured_dims is None:
      measured_dims = list(range(ndim_state))
    elif not isinstance(measured_dims, list):
      measured_dims = [measured_dims]
    self.measured_dims = measured_dims
    self.ndim = self.ndim_meas = len(measured_dims)
    self.np_random = np.random.RandomState(seed)
    
  def __call__(self, state: np.array, noise: bool = True):
    H = self.matrix()
    noise = self.sample_noise() if noise else 0
    return H @ state + noise
  
  def matrix(self):
    # NOTE: For now, using binary matrix with ones at measured inds
    H = np.zeros((self.ndim_meas, self.ndim_state))
    H[np.arange(self.ndim_meas), self.measured_dims] = 1
    return H
  
  def covar(self):
    return self.noise_covar
  
  def sample_noise(self):
    noise = self.np_random.multivariate_normal(
      mean=np.zeros(self.ndim), cov=self.noise_covar)
    return noise
