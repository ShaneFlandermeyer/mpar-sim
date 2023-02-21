import pytest
from mpar_sim.common.coordinate_transform import sph2cart_covar
import numpy as np


def test_sph2cart_covar():
  sph_covar = np.diag([9, 6.25, 4, 1])
  az = 45
  el = -10
  r = 1000
  actual_pos_covar, actual_vel_covar = sph2cart_covar(sph_covar, az, el, r)
  expected_pos_covar = 1e3*np.array(
      [[1.3601,   -1.2988,    0.2297],
       [-1.2988,    1.3601,    0.2297],
          [0.2297,    0.2297,    1.8466, ]]
  )
  assert np.allclose(expected_pos_covar, actual_pos_covar, atol=1e-1)


if __name__ == '__main__':
  pytest.main()
