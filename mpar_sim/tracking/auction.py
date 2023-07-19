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
  is_assigned = object_inds != None
  if np.any(is_assigned):
    agent_inds = agent_inds[is_assigned]
    object_inds = object_inds[is_assigned].astype(int)
    happy[agent_inds] = np.max(cost_mat[agent_inds] - prices, axis=1) - \
      (cost_mat[agent_inds, object_inds] - prices[object_inds]) <= eps
      
  return happy


def auction(
    a: np.ndarray,
    eps: float,
) -> Tuple[List[Tuple], np.ndarray]:

  n_object, n_agent = a.shape
  assert n_object >= n_agent, "Number of agents must be greater than number of objects."

  # Initialize all observations as unassociated
  agents = np.arange(n_agent)
  assigned_objects = np.array([None]*n_agent)
  happy = np.zeros(n_agent, dtype=bool)
  prices = np.zeros(n_object)
  iter = 0
  
  while True:
    # Check if all agents are happy
    happy = is_happy(cost_mat=a,
                     prices=prices,
                     agent_inds=agents,
                     object_inds=assigned_objects,
                     eps=eps)
    if np.all(happy):
      break
    iagent = np.where(~happy)[0][0]
    
    
    
    # If another agent already has the desired object, swap it. Otherwise, assign it directly
    iobject = np.argmax(a[iagent] - prices)    
    if iobject in assigned_objects:
      iswap = np.where(assigned_objects == iobject)[0].item()
      assigned_objects[iswap], assigned_objects[iagent] = \
          assigned_objects[iagent], assigned_objects[iswap]
    else:
      assigned_objects[iagent] = iobject
      
    # Update the price of the assigned object
    mask = np.arange(n_object) != iobject
    yj = a[iagent, iobject] - prices[iobject] - \
        np.max(a[iagent][mask] - prices[mask])
    prices[iobject] += yj + eps
    
    iter += 1
    

  print(f"Converged after {iter} iterations.")
  # Return a list of assignment pairs and the computed prices for each object
  return [(a, o) for a, o in zip(agents, assigned_objects)], prices
