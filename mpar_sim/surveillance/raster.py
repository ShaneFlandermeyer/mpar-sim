import numpy as np
from typing import List, Tuple

from mpar_sim.common.coordinate_transform import azel2uv, uv2azel


def uniform_rectangular_grid(xlims: List[int],
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
  # To account for the fact that the dx may not divide evenly into the x-span, we need to add a small offset to the axis limits to shrink it so it fits

  # Compute the x-axis values
  x_span = xlims[1] - xlims[0]
  nx = int(x_span // dx)
  x_offset = 0.5*(nx * dx - x_span)
  x = np.linspace(xlims[0] - x_offset, xlims[1] + x_offset, nx+1)

  # Compute the y-axis values
  y_span = ylims[1] - ylims[0]
  ny = int(y_span // dy)
  y_offset = 0.5*(ny * dy - y_span)
  y = np.linspace(ylims[0] - y_offset, ylims[1] + y_offset, ny+1)

  xgrid, ygrid = np.meshgrid(x, y)

  return xgrid, ygrid


def raster_grid_points(az_lims: List[float],
                   el_lims: List[float],
                   az_beamwidth: float,
                   el_beamwidth: float,
                   az_spacing_beamwidths: float,
                   el_spacing_beamwidths: float) -> Tuple[np.ndarray, np.ndarray]:
  """
  Compute a raster scan grid that is uniformly spaced in UV space. This way, the beam pattern does not vary with off-boresight scan angle. See Klemm2017.
  This grid is triangular such that successive rows are offset by half a beamwidth.
  
  Parameters
  ----------
  az_lims : List[float]
      Scan limits in azimuth
  el_lims : List[float]
      Scan limits in elevation
  az_beamwidth : float
      Azimuth beamwidth
  el_beamwidth : float
      Elevation beamwidth
  az_spacing_beamwidths : float
      Beamwidths between grid points in the azimuth direction
  el_spacing_beamwidths : float
      Beamwidths between grid points in the elevation direction  
  Returns
  -------
  Tuple[np.ndarray, np.ndarray]
      A tuple containing two arrays:
        - The azimuth points of the grid (in degrees)
  """
  az_spacing = az_beamwidth * az_spacing_beamwidths
  el_spacing = el_beamwidth * el_spacing_beamwidths
  uv_spacing = azel2uv(az_spacing, el_spacing)

  # Compute a uniform grid in u and v space
  ulim = azel2uv(az_lims, el_lims[1])[0]
  vlim = azel2uv(az_lims[0], el_lims)[1]
  dx = uv_spacing[0]/2
  dy = np.sqrt(uv_spacing[1]**2 - 0.25*uv_spacing[0]**2)
  ugrid, vgrid = uniform_rectangular_grid(ulim, vlim, uv_spacing[0], dy)

  # Offset every other row by half the normal spacing to make the lattice triangular
  ugrid[1::2] += dx

  # Remove points that are now outside the scan limits due to the offset applied above
  valid = ugrid < ulim[1]
  ugrid = ugrid[valid]
  vgrid = vgrid[valid]

  # UV pairs must also have a magnitude less than 1
  hypot = np.hypot(ugrid, vgrid)
  ugrid = ugrid[hypot < 1]
  vgrid = vgrid[hypot < 1]

  # Convert back to az/el and remove the points outside the scan limits
  az_points, el_points = uv2azel(ugrid, vgrid, degrees=True)

  # Remove points that are outside the scan limits after conversion
  valid_az = (az_points >= az_lims[0]) & (az_points <= az_lims[1])
  valid_el = (el_points >= el_lims[0]) & (el_points <= el_lims[1])
  az_points = az_points[valid_az & valid_el]
  el_points = el_points[valid_az & valid_el]

  return az_points, el_points
