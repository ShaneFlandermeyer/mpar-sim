import numpy as np

def block_diag(mat: np.array,
               nreps: int = 1,
               ) -> np.array:
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
  for i in range(nreps):
    result[i*rows:(i+1)*rows, i*cols:(i+1)*cols] = mat

  return result


def jacobian(func, x, **kwargs):
  """Compute Jacobian through finite difference calculation

    Parameters
    ----------
    fun : function handle
        A (non-linear) transition function
        Must be of the form "y = fun(x)", where y can be a scalar or \
        :class:`numpy.ndarray` of shape `(Nd, 1)` or `(Nd,)`
    x : :class:`numpy.ndarray`
        A state vector of shape `(Ns, 1)`

    Returns
    -------
    jac: :class:`numpy.ndarray` of shape `(Nd, Ns)`
        The computed Jacobian
  """
  ndim, _ = np.shape(x)

  # For numerical reasons the step size needs to large enough. Aim for 1e-8
  # relative to spacing between floating point numbers for each dimension
  delta = 1e8*np.spacing(x.astype(np.double).ravel())
  delta[delta < 1e-8] = 1e-8
  x2 = np.tile(x, ndim+1) + np.eye(ndim, ndim+1)*delta[:, np.newaxis]
  F = func(x2, **kwargs)
  jac = (F[:, :ndim] - F[:, -1:]) / delta
  return jac.astype(np.double)
