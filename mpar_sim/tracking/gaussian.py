from typing import List, Tuple
import numpy as np
from scipy.stats import multivariate_normal

def mix_gaussians(means: List[np.ndarray],
                  covars: List[np.ndarray],
                  weights: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
  """
  Compute a Gaussian mixture as a weighted sum of N Gaussian distributions, each with dimension D.

  Parameters
  ----------
  means : np.ndarray
      Length-N list of D-dimensional arrays of mean values for each Gaussian components
  covars : np.ndarray
      Length-N list of D x D covariance matrices for each Gaussian component
  weights : np.ndarray
      Length-N array of weights for each component

  Returns
  -------
  Tuple[np.ndarray, np.ndarray]
    Mixture PDF mean and covariance
      
  """
  assert len(means) == len(covars) == len(weights)
  
  N = len(weights)
  x = np.atleast_2d(means)
  P = np.atleast_3d(covars)
  mix_mean = np.dot(weights, x)
  mix_covar = np.zeros_like(P[0])
  for i in range(N):
    mix_covar += weights[i] * (P[i] + x[i] @ x[i].T)
  mix_covar -= mix_mean @ mix_mean.T
  
  return mix_mean, mix_covar