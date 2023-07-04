import pytest
from mpar_sim.surveillance.volume import VolumeSearchManager, raster_grid_points
import numpy as np


class TestVolumeSearchManager():
  @pytest.fixture()
  def manager(self):
    return VolumeSearchManager(
        az_beamwidth=8,
        el_beamwidth=10,
        az_lims=[-60, 60],
        el_lims=[-30, 0],
        az_spacing_beamwidths=0.85,
        el_spacing_beamwidths=0.85,
    )

  def test_init(self, manager: VolumeSearchManager):
    assert len(manager.az_points) == 56
    assert len(manager.el_points) == 56

  def test_generate_looks(self, manager: VolumeSearchManager):
    time = 0
    looks = manager.generate_looks(time)

    # Check the results
    assert len(looks) == 1
    assert looks[0].start_time == time
    assert np.allclose(
        looks[0].azimuth_steering_angle, -52.03464462683514, atol=1e-4)
    assert np.allclose(
        looks[0].elevation_steering_angle, -26.973393595542024, atol=1e-4)

    # Do another look to ensure the position increments as expected
    time += 1
    looks = manager.generate_looks(time)
    assert len(looks) == 1
    assert looks[0].start_time == time
    assert np.allclose(
        looks[0].azimuth_steering_angle, -41.070390641104545, atol=1e-4)
    assert np.allclose(
        looks[0].elevation_steering_angle, -26.973393595542024, atol=1e-4)


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
