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


def is_happy(cost_mat: np.ndarray,
             prices: np.ndarray,
             agent_inds: np.ndarray,
             object_inds: np.ndarray,
             eps: float):
  happy = np.zeros(len(agent_inds), dtype=bool)
  if np.any(object_inds != None):
    valid_inds = np.where(object_inds != None)[0]
    agent_inds = agent_inds[valid_inds]
    object_inds = object_inds[valid_inds].astype(int)
    happy[agent_inds] = np.max(cost_mat[agent_inds] - prices, axis=1) - \
      (cost_mat[agent_inds, object_inds] - prices[object_inds]) < eps
      
  return happy


def auction(
    a: np.ndarray,
    eps: float,
    maxiter: Optional[int] = None,
) -> Tuple[List[Tuple], np.ndarray]:

  n_object, n_agent = a.shape
  assert n_object >= n_agent, "Number of agents must be greater than number of objects."

  # Initialize all observations as unassociated
  agents = np.arange(n_agent)
  assigned_objects = [None for _ in range(n_agent)]
  happy = np.zeros(n_agent, dtype=bool)
  prices = np.zeros(n_object)
  iter = 0
  while True:
    iagent = np.where(~happy)[0][0]
    
    # Exchange favored object with the person currently assigned to it
    iobj = np.argmax(a[iagent] - prices)    
    if iobj in assigned_objects:
      # Swap the assigned objects
      iswap = np.where(assigned_objects == iobj)[0].item()
      assigned_objects[iswap], assigned_objects[iagent] = \
          assigned_objects[iagent], assigned_objects[iswap]
    else:
      assigned_objects[iagent] = iobj

    # Update the price of the assigned objects
    mask = np.arange(n_object) != iobj
    yj = a[iagent, iobj] - prices[iobj] - \
        np.max(a[iagent][mask] - prices[mask])
    prices[iobj] += yj + eps
    iter += 1
    
    happy = is_happy(cost_mat=a,
                     prices=prices,
                     agent_inds=agents,
                     object_inds=np.array(assigned_objects),
                     eps=eps)
    if np.all(happy):
      break


  # Sort the assigned array based on the first column
  # assigned = assigned[np.argsort(assigned[:, 0])]

  return [(a, o) for a,o in zip(agents, assigned_objects)], prices
