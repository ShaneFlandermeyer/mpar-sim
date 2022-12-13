import numpy as np

def block_diag(mat: np.ndarray, nreps: int = 1) -> np.ndarray:
  """
  Create a block diagonal matrix from a 2D array, where the input array is repeated nrep times

  Parameters
  ----------
  arr : np.ndarray
      Array to repeat
  nreps : int, optional
      Number of repetitions of the matrix, by default 1

  Returns
  -------
  np.ndarray
      A block diagonal matrix
  """
  rows, cols = mat.shape
  result = np.zeros((nreps * rows, nreps * cols), dtype=mat.dtype)
  for k in range(nreps):
    result[k*rows:(k+1)*rows, k*cols:(k+1)*cols] = mat
  return result

def constant_velocity(state: np.ndarray, q: float, dt: float):
  """
  3D constant velocity model with white noise acceleration

  Parameters
  ----------
  state : np.ndarray
      State vector
  q : float
      Velocity noise diffusion coefficient
  dt : float
      Time interval

  Returns
  -------
  np.ndarray
      State vector
  """
  # State transition matrix
  F = np.array([[1, dt],
                [0, 1]])
  F = block_diag(F, nreps=3)
  # Process noise matrix
  Q = np.array([[dt**3/3, dt**2/2],
                [dt**2/2, dt]]) * q
  Q = block_diag(Q, nreps=3)
  return F @ state + np.random.multivariate_normal(np.zeros(6), Q)[:, np.newaxis]
