from typing import List, Tuple
import numpy as np

from mpar_sim.common.coordinate_transform import azel2uv, uv2azel


def cued_search_grid(az_center: float,
                     el_center: float,
                     az_beamwidth: float,
                     el_beamwidth: float,
                     az_spacing_beamwidths: float,
                     el_spacing_beamwidths: float,
                     az_lims: List[float] = None,
                     el_lims: List[float] = None,
                     ) -> Tuple[np.ndarray, np.ndarray]:
  # TODO: Write a test that compares this to the matlab function
  # Convert spacing parameters to UV space
  az_spacing = az_beamwidth * az_spacing_beamwidths
  el_spacing = el_beamwidth * el_spacing_beamwidths
  u_spacing, v_spacing = azel2uv(az_spacing, el_spacing)

  # Compute the cued search grid points. This grid forms a triangular lattice in UV space
  du = u_spacing / 2
  dv = np.sqrt(v_spacing**2 - du**2)
  u_grid = np.array([-u_spacing, 0, u_spacing,
                     -u_spacing + du, du, -u_spacing + du, du])
  v_grid = np.array([0, 0, 0, dv, dv, -dv, -dv])

  # Center the grid points on the look direction
  u_center, v_center = azel2uv(az_center, el_center)
  u_points = u_center + u_grid
  v_points = v_center + v_grid

  # UV pairs must have magnitdude <= 1
  hypot = np.hypot(u_points, v_points)
  u_points = u_points[hypot <= 1]
  v_points = v_points[hypot <= 1]

  # Remove points that are outside the scan limits
  az_points, el_points = uv2azel(u_points, v_points, degrees=True)
  if az_lims is not None:
    valid_az = (az_points >= az_lims[0]) & (az_points <= az_lims[1])
  else:
    valid_az = np.ones_like(az_points, dtype=bool)
  if el_lims is not None:
    valid_el = (el_points >= el_lims[0]) & (el_points <= el_lims[1])
  else:
    valid_el = np.ones_like(el_points, dtype=bool)
  az_points = az_points[valid_az & valid_el]
  el_points = el_points[valid_az & valid_el]

  return az_points, el_points
