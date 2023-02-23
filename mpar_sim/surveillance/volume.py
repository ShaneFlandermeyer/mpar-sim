import datetime
from typing import List, Optional, Tuple, Union

import numpy as np

from mpar_sim.common.coordinate_transform import azel2uv, uv2azel
from mpar_sim.types.look import Look


class VolumeSearchManager():
  def __init__(self,
               # Required parameters
               az_beamwidth: float,
               el_beamwidth: float,
               az_lims: List[float],
               el_lims: List[float],
               # Task parameters
               center_frequency: Optional[float] = None,
               bandwidth: Optional[float] = None,
               pulsewidth: Optional[float] = None,
               prf: Optional[float] = None,
               n_pulses: Optional[int] = None,
               # Raster scan parameters
               az_spacing_beamwidths: float = 0.85,
               el_spacing_beamwidths: float = 0.85,
               ):
    """
    This object performs a volume search with a raster scan pattern that is uniformly spaced in UV coordinates. 
        
    Parameters
    ----------
    az_beamwidth : float
        Azimuth beamwidth
    el_beamwidth : float
        Elevation beamwidth
    az_lims : List[float]
        Scan limits in azimuth
    el_lims : List[float]
        Scan limits in elevation
    center_frequency : Optional[float], optional
        The center frequency of the radar, by default None
    bandwidth : Optional[float], optional
        The bandwidth of the radar, by default None
    pulsewidth : Optional[float], optional
        The pulsewidth of the radar, by default None
    prf : Optional[float], optional
        The pulse repetition frequency of the radar, by default None
    n_pulses : Optional[int], optional
        The number of pulses in the volume, by default None
    az_spacing_beamwidths : float, optional
        Beamwidths between grid points in the azimuth direction, by default 0.85
    el_spacing_beamwidths : float, optional
        Beamwidths between grid points in the elevation direction, by default 0.85
    """
    # Parameter checks
    assert az_lims[1] > az_lims[0], "Azimuth limits must be in ascending order"
    assert el_lims[1] > el_lims[0], "Elevation limits must be in ascending order"

    # Required parameters
    self.az_beamwidth = az_beamwidth
    self.el_beamwidth = el_beamwidth
    self.az_lims = az_lims
    self.el_lims = el_lims
    # Task parameters
    self.center_frequency = center_frequency
    self.bandwidth = bandwidth
    self.pulsewidth = pulsewidth
    self.prf = prf
    self.n_pulses = n_pulses
    # Raster scan parameters
    self.az_spacing_beamwidths = az_spacing_beamwidths
    self.el_spacing_beamwidths = el_spacing_beamwidths

    # Compute the grid points
    self.az_points, self.el_points = raster_grid_points(
        az_lims=az_lims,
        el_lims=el_lims,
        az_beamwidth=az_beamwidth,
        el_beamwidth=el_beamwidth,
        az_spacing_beamwidths=az_spacing_beamwidths,
        el_spacing_beamwidths=el_spacing_beamwidths
    )
    self.n_positions = len(self.az_points)
    self.position_index = 0

  def generate_looks(self,
                     time: Union[float, datetime.datetime]
                     ) -> List[Look]:
    looks = []
    # Create a look for the next beam position
    look = Look(
        azimuth_steering_angle=self.az_points[self.position_index],
        elevation_steering_angle=self.el_points[self.position_index],
        azimuth_beamwidth=self.az_beamwidth,
        elevation_beamwidth=self.el_beamwidth,
        center_frequency=self.center_frequency,
        bandwidth=self.bandwidth,
        pulsewidth=self.pulsewidth,
        prf=self.prf,
        n_pulses=self.n_pulses,
        start_time=time,
    )
    looks.append(look)

    # Increment the raster scan position
    self.position_index = (self.position_index + 1) % self.n_positions
    return looks


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
        - The elevation points of the grid (in degrees)
  """
  az_spacing = az_beamwidth * az_spacing_beamwidths
  el_spacing = el_beamwidth * el_spacing_beamwidths
  uv_spacing = azel2uv(az_spacing, el_spacing)

  # Compute a uniform grid in u and v space
  ulim = azel2uv(az_lims, el_lims[1])[0]
  vlim = azel2uv(az_lims[0], el_lims)[1]
  dx = uv_spacing[0]/2
  dy = np.sqrt(uv_spacing[1]**2 - dx**2)
  u_grid, v_grid = uniform_rectangular_grid(ulim, vlim, uv_spacing[0], dy)

  # Offset every other row by half the normal spacing to make the lattice triangular
  u_grid[1::2] += dx

  # Remove points that are now outside the scan limits due to the offset applied above
  valid = u_grid < ulim[1]
  u_grid = u_grid[valid]
  v_grid = v_grid[valid]

  # UV pairs must also magnitude <= 1
  hypot = np.hypot(u_grid, v_grid)
  u_grid = u_grid[hypot <= 1]
  v_grid = v_grid[hypot <= 1]

  # Convert back to az/el and remove the points outside the scan limits
  az_points, el_points = uv2azel(u_grid, v_grid, degrees=True)

  # Remove points that are outside the scan limits after conversion
  valid_az = (az_points >= az_lims[0]) & (az_points <= az_lims[1])
  valid_el = (el_points >= el_lims[0]) & (el_points <= el_lims[1])
  az_points = az_points[valid_az & valid_el]
  el_points = el_points[valid_az & valid_el]

  return az_points, el_points


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
