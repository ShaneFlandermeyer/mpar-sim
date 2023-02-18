import pytest
from mpar_sim.surveillance.raster import raster_grid_points
import numpy as np


def test_raster_grid():
  az_lims = [-60, 60]
  el_lims = [-30, 0]
  az_bw = 8
  el_bw = 10
  az_spacing = el_spacing = 0.85

  az_points, el_points = raster_grid_points(az_lims, el_lims, az_bw,
                           el_bw, az_spacing, el_spacing)
  
  assert az_points.shape == (56,)
  assert el_points.shape == (56,)
  
  expected_min = np.array([-56.4775, -26.9734])
  actual_min = np.array([np.min(az_points), np.min(el_points)])
  assert np.allclose(actual_min, expected_min, atol=1e-4)
  
  expeted_max = np.array([np.max(az_points), np.max(el_points)])
  actual_max = np.array([56.4775, -2.6608])
  assert np.allclose(actual_max, expeted_max, atol=1e-4)


if __name__ == '__main__':
  pytest.main()
