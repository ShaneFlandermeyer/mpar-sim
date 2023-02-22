import pytest
from mpar_sim.surveillance.cued import cued_search_grid, CuedSearchManager
import numpy as np

from mpar_sim.types.detection import Detection


class TestCuedSearchManager():
  @pytest.fixture
  def manager(self):
    return CuedSearchManager(
        az_beamwidth=4,
        el_beamwidth=5,
        az_lims=[-60, 60],
        el_lims=[-30, 0],
        az_spacing_beamwidths=0.85,
        el_spacing_beamwidths=0.85,
    )

  def test_process_detections(self, manager: CuedSearchManager):
    detections = [Detection(
        state_vector=np.array([-20, -12, 10e3, 0]),
    )]
    manager.process_detections(detections)

    assert manager.beam_positions_az.shape == (7,)
    assert manager.beam_positions_el.shape == (7,)


def test_raster_grid():
  az_lims = [-60, 60]
  el_lims = [-30, 0]
  az_bw = 4
  el_bw = 5
  az_spacing = el_spacing = 0.85
  az_center = -20
  el_center = -12

  az_points, el_points = cued_search_grid(
      az_center, el_center, az_bw, el_bw, az_spacing, el_spacing, *az_lims, *el_lims)

  assert az_points.shape == (7,)
  assert el_points.shape == (7,)

  expected_min = np.array([-23.7336, -16.0135])
  actual_min = np.array([min(az_points), min(el_points)])
  assert np.allclose(actual_min, expected_min, atol=1e-4)

  actual_max = np.array([max(az_points), max(el_points)])
  expected_max = np.array([-16.3530, -8.0454])
  assert np.allclose(actual_max, expected_max, atol=1e-4)


if __name__ == '__main__':
  pytest.main()
