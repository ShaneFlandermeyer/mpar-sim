from mpar_sim.models.measurement.nonlinear import CartesianToRangeAzElRangeRate
import pytest
import numpy as np


class TestCartesianToAzElBearingRange:
  model = CartesianToRangeAzElRangeRate()
  model.discretize_measurements = False
  model.alias_measurements = False

  def test_conversion(self):
    """Test conversion function from Cartesian to range, azimuth, elevation, and range rate"""
    state = np.array([1000, 0, 1000, 100, 0, 0])
    r = np.sqrt(state[0]**2 + state[2]**2 + state[4]**2)
    el = np.rad2deg(np.arcsin(state[4]/r))
    az = np.rad2deg(np.arctan2(state[2], state[0]))
    rr = np.dot(state[self.model.position_mapping],
                state[self.model.velocity_mapping]) / r
    expected = np.array([el, az, r, rr])[:, np.newaxis]

    actual = self.model.function(state, noise=False)

    assert np.all(expected == actual)

  # def test_jacobian(self):
  #   model = CartesianToRangeAzElRangeRate()
  #   state = np.ones((6,1))
  #   J = model.jacobian(state)
  #   assert False


if __name__ == '__main__':
  pytest.main()
