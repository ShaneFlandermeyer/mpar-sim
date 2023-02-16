import numpy as np
from typing import List


def uniform_grid_points(xlims: List[int],
                ylims: List[int],
                dx: float,
                dy: float,
                ) -> np.ndarray:
  """
  Compute a list of points that lie on a uniform grid with the given specifications

  Parameters
  ----------
  xlims : List[int]
      Limits of the grid axis in the x-dimension
  ylims : List[int]
      Limits of the grid axis in the y-dimesnion
  dx : float
      Spacing between points in the x-dimension
  dy : float
      Spacing between points in the y-dimension

  Returns
  -------
  np.ndarray
      An N x 2 array of grid points, where the first column contains the x-axis points and the second contains the y-axis points
  """
  xaxis = np.arange(xlims[0], xlims[1], dx)
  yaxis = np.arange(ylims[0], ylims[1], dy)
  xgrid, ygrid = np.meshgrid(xaxis, yaxis)
  return np.stack((xgrid.flatten(), ygrid.flatten()), axis=1)