import numpy as np
from typing import Tuple
import scipy

def merwe_scaled_sigma_points(x: np.ndarray,
                              P: np.ndarray,
                              # Merwe parameters
                              alpha: float,
                              beta: float,
                              kappa: float
                              ) -> Tuple[np.ndarray, ...]:
  """
  Compute sigma points (and the weights for each point) using Van der Merwe's algorithm

  Parameters
  ----------
  mean : np.ndarray
      Input mean
  covar : np.ndarray
      Input covariance
  alpha : float
      Van der Merwe alpha parameter
  beta : float
      Van der Merwe beta parameter
  kappa : float
      Van der Merwe kappa parameter

  Returns
  -------
  Tuple[np.ndarray, np.ndarray, np.ndarray]
      A tuple containing:
        - An array where each COLUMN contains the position of a sigma point. This is because the numpy cholesky function returns a lower triangular matrix by default.
        - An array containing the weights for each sigma point mean
        - An array containing the weights for each sigma point covariance
  """
  # Compute the sigma points
  n = x.size
  lambda_ = alpha**2 * (n + kappa) - n
  U = scipy.linalg.cholesky((n + lambda_) * P)

  x = x[np.newaxis, :] if x.ndim == 1 else x
  sigmas = np.concatenate([x, x - U, x + U], axis=0)

  # Compute the weight for each point
  Wc = Wm = np.full(2*n+1, 1 / (2*(n+lambda_)))
  Wc[0] = lambda_ / (n + lambda_) + (1 - alpha**2 + beta)
  Wm[0] = lambda_ / (n + lambda_)

  return sigmas, Wm, Wc