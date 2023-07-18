from mpar_sim.tracking.auction import auction
import pytest
import numpy as np


def test_auction():
  eps = 0.2
  ass_mat = np.array([[1, 3], [10, 15]])
  a, p = auction(ass_mat, eps)
  assert a == [(0, 0), (1, 1)]
  assert np.allclose(p, [0, 5.2])


if __name__ == '__main__':
  pytest.main()
