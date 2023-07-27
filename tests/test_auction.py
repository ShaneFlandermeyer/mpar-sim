from mpar_sim.tracking.auction import auction
import pytest
import numpy as np


def test_auction():
  A = np.array([[10, 5, 8, 9],
                [7, -1, 20, -1],
                [-1, 21, -1, -1],
                [-1, 15, 17, -1],
                [-1, -1, 16, 22]])
  # Create a mask for the above, with the x elements zeroed out
  mask = 1 - np.array([[1, 1, 1, 1],
                       [1, 0, 1, 0],
                       [0, 1, 0, 0],
                       [0, 1, 1, 0],
                       [0, 0, 1, 1]])
  A = np.ma.masked_array(data=A, mask=mask)
  # ass_mat = np.array([[1, 3], [10, 15]])
  eps = 1 / np.min(A.shape)
  a, p = auction(A, eps, maximize=False)
  total_cost = sum(A[i, j] for i, j in a)
  assert a == [(0, 3), (1, 0), (3, 1), (4, 2)]
  assert total_cost == 47


if __name__ == '__main__':
  pytest.main()
