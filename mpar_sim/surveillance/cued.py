import datetime
from functools import lru_cache
from typing import List, Tuple, Union

import numpy as np

from mpar_sim.common.coordinate_transform import azel2uv, uv2azel
from mpar_sim.types.detection import Detection
from mpar_sim.types.look import Look


class CuedSearchManager():
  def __init__(self,
               az_beamwidth: float,
               el_beamwidth: float,
               az_lims: List[float],
               el_lims: List[float],
               az_spacing_beamwidths: float = 0.75,
               el_spacing_beamwidths: float = 0.75,
               ):
    self.az_beamwidth = az_beamwidth
    self.el_beamwidth = el_beamwidth
    self.az_lims = az_lims
    self.el_lims = el_lims
    self.az_spacing_beamwidths = az_spacing_beamwidths
    self.el_spacing_beamwidths = el_spacing_beamwidths

    # Store a queue of beam positions to search
    self.beam_positions_az = []
    self.beam_positions_el = []

  def process_detections(self, detections: List[Detection]) -> None:
    """
    Compute the cued search grid points for each detection and add them to the queue of beam positions to search.

    Parameters
    ----------
    detections : List[Detection]
        Detections to process.
    """

    # TODO: Add a clustering algorithm such as DBSCAN to group detections that likely came from the same target.

    for detection in detections:
      measurement = detection.state_vector
      az_points, el_points = cued_search_grid(
          az_center=measurement[0],
          el_center=measurement[1],
          az_beamwidth=self.az_beamwidth,
          el_beamwidth=self.el_beamwidth,
          az_spacing_beamwidths=self.az_spacing_beamwidths,
          el_spacing_beamwidths=self.el_spacing_beamwidths,
          min_az=self.az_lims[0],
          max_az=self.az_lims[1],
          min_el=self.el_lims[0],
          max_el=self.el_lims[1],
      )

      # Add the current beam positions to the queue
      self.beam_positions_az = np.concatenate(
          (self.beam_positions_az, az_points))
      self.beam_positions_el = np.concatenate(
          (self.beam_positions_el, el_points))

  def generate_looks(self,
                     time: Union[float, datetime.datetime]
                     ) -> List[Look]:
    pass  # TODO


@lru_cache
def cued_search_grid(az_center: float,
                     el_center: float,
                     az_beamwidth: float,
                     el_beamwidth: float,
                     az_spacing_beamwidths: float,
                     el_spacing_beamwidths: float,
                     min_az: float = None,
                     max_az: float = None,
                     min_el: float = None,
                     max_el: float = None,
                     ) -> Tuple[np.ndarray, np.ndarray]:
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
  valid_az = np.ones_like(az_points, dtype=bool)
  valid_el = np.ones_like(el_points, dtype=bool)
  if min_az:
    valid_az = valid_az & (az_points >= min_az)
  if max_az:
    valid_az = valid_az & (az_points <= max_az)
  if min_el:
    valid_el = valid_el & (el_points >= min_el)
  if max_el:
    valid_el = valid_el & (el_points <= max_el)

  az_points = az_points[valid_az & valid_el]
  el_points = el_points[valid_az & valid_el]

  return az_points, el_points
