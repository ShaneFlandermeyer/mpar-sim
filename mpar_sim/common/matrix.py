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