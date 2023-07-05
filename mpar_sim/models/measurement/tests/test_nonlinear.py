from mpar_sim.models.measurement.nonlinear import CartesianToRangeAzElVelocity
import pytest
import numpy as np

from mpar_sim.types.detection import Detection


class TestCartesianToAzElBearingRange:
  model = CartesianToRangeAzElVelocity()
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
    expected = np.array([az, el, r, rr])

    actual = self.model(state, noise=False)

    assert np.allclose(expected, actual)

  def test_jacobian(self):
    state = np.array([1, 10, 2, 20, 3, 30])
    actual = self.model.jacobian(state)
    expected = np.array(
        [[-22.9183, 0, 11.4592, 0, 0, 0],
         [-5.4907, 0, -10.9815, 0, 9.1512, 0],
            [0.2673, 0, 0.5345, 0, 0.8018, 0],
            [0, 0.2673, 0, 0.5345, 0, 0.8018]
         ])
    assert np.allclose(expected, actual, atol=1e-4)

  def test_inverse_function(self):
    """
    Expected results from the example here:
    https://www.mathworks.com/help/radar/ref/initcvekf.html
    """
    mm = CartesianToRangeAzElVelocity(
        translation_offset=np.array([25, -40, 0]),
        velocity=np.array([0, 5, 0])
    )
    detection = Detection(measurement=np.array([45, -10, 1000, -4]))
    actual = mm.inverse_function(detection.measurement)
    expected = np.array(
        [721.3642, -2.7855, 656.3642, 2.2145, -173.6482, 0.6946])
    assert np.allclose(expected.ravel(), actual.ravel(), atol=1e-4)
    


if __name__ == '__main__':
  pytest.main()
