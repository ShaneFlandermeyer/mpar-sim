from typing import List, Union

import numpy as np


class LinearMeasurementModel():
  def __init__(self,
               ndim_state: int,
               covar: np.ndarray,
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

  def __call__(self,
               state: Union[np.ndarray, List[np.ndarray]],
               noise: bool = True):
    if isinstance(state, list):
      n_inputs = len(state)
    else:
      n_inputs = 1

    state = np.array(state).T
    H = self.matrix()
    deterministic = H @ state
    out = deterministic.T
    if noise:
      noise = np.array([self.sample_noise()
                     for _ in range(n_inputs)]).T
      out += noise.reshape(out.shape)

    return list(out) if n_inputs > 1 else out

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
