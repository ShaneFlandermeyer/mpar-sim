"""
Algorithm steps:

1. Initialize all observations as unassigned. Initialize track prices to a small (or zero) value.
2. Select an unassigned observation j. If none exists, terminate.
3. Find the "best" track for each observation j.
4. Unassign the observation previously assigned to i_j and assign track i_j to observation j
5. Set the price of i_j to the level at which the observation j is almost happy.
6. Return to step 2. 
"""
from typing import List, Optional, Tuple
import numpy as np
from collections import deque


def auction(
    assignment_matrix: np.ndarray,
    eps: float,
    maxiter: Optional[int] = None,
) -> Tuple[List[Tuple], np.ndarray]:

  nobs, ntrack = assignment_matrix.shape

  # Initialize all observations as unassociated
  mask = np.ones(ntrack, dtype=bool)
  prices = np.zeros(ntrack)
  unassigned = deque([i for i in range(nobs)])
  assignments = []

  iter = 0
  while True:
    obs_ind = unassigned.popleft()
    # Find the best track for the current observation
    track_ind = np.argmax(assignment_matrix[obs_ind] - prices)
    # Unassign the observation previously assigned to ij (if any) and assign ij to j
    unassigned.extend([o for o, t in assignments if t == track_ind])
    assignments = list(filter(lambda x: x[1] != track_ind, assignments))
    assignments.append((obs_ind, track_ind))
    if len(unassigned) == 0:
      break

    # Update the price of the assigned track
    mask[:] = True
    mask[track_ind] = False
    yj = assignment_matrix[obs_ind][track_ind] - \
        prices[track_ind] - np.max(assignment_matrix[obs_ind][mask])
    prices[track_ind] += yj + eps

    iter += 1
    if maxiter is not None and iter >= maxiter:
      break

  # Sort assignments by observation index
  assignments = sorted(assignments, key=lambda x: x[0])
  return assignments, prices