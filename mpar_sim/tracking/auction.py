"""
Algorithm steps:

1. Initialize all observations as unassigned. Initialize track prices to a small (or zero) value.
2. Select an unassigned observation j. If none exists, terminate.
3. Find the "best" track for each observation j.
4. Unassign the observation previously assigned to i_j and assign track i_j to observation j
5. Set the price of i_j to the level at which the observation j is almost happy.
6. Return to step 2.
"""
from typing import List, Tuple
import numpy as np


def is_happy(cost_mat: np.ndarray,
             prices: np.ndarray,
             agent_inds: np.ndarray,
             object_inds: np.ndarray,
             eps: float) -> np.ndarray:
  """
  Computes whether each agent in a list is almost happy with its current assignment, as defined by Bertsekas1989 equation (7).

  Parameters
  ----------
  cost_mat : np.ndarray
      Cost matrix 
  prices : np.ndarray
      Array of prices for each object
  agent_inds : np.ndarray
      Cost matrix row indices for each agent
  object_inds : np.ndarray
      Cost matrix column indices for each object
  eps : float
      epsilon-complimentary slackness variable

  Returns
  -------
  np.ndarray
      A boolean array that is true for (almost) happy agents, and false otherwise.
  """
  happy = np.zeros(len(agent_inds), dtype=bool)
  is_assigned = object_inds != None
  if np.any(is_assigned):
    agent_inds = agent_inds[is_assigned]
    object_inds = object_inds[is_assigned].astype(int)
    happy[agent_inds] = np.max(cost_mat[agent_inds] - prices, axis=1) - \
        (cost_mat[agent_inds, object_inds] - prices[object_inds]) <= eps

  return happy


def auction(
    cost_mat: np.ndarray,
    eps: float,
    maximize: bool = True,
) -> Tuple[List[Tuple], np.ndarray]:
  """
  Computes the optimal assignment of agents to objects using the auction algorithm.

  Parameters
  ----------
  cost_mat : np.ndarray
      Cost matrix
  eps : float
      Epsilon-complimentary slackness variable
  maximize : bool, optional
      If true, the auction attempts to maximize the sum of costs in its feasible assignment. If False, it minimizes cost, by default True

  Returns
  -------
  Tuple[List[Tuple], np.ndarray]
    - A list of tuples containing assignment pairs
    - An array containing the computed prices for each object
  """

  n_agent, n_object = cost_mat.shape
  # Convert masked cost matrix to a reguar numpy array
  if np.ma.is_masked(cost_mat):
    cost_mat = cost_mat.filled(np.ma.maximum_fill_value(cost_mat))
  # The following code assumes that there are more objects than agents, and that agents/objects are the rows/columns of the cost matrix. If this is not the case, we transpose the cost matrix, solve the optimization, then put the resulting pairs back in the original order.
  transpose = False if n_object >= n_agent else True
  if transpose:
    cost_mat = cost_mat.T
    n_agent, n_object = n_object, n_agent
  if not maximize:
    cost_mat = -cost_mat

  # Initialize all objects as unassociated, all agents as unhappy, and all prices as zero
  agents = np.arange(n_agent)
  assigned_objects = np.array([None]*n_agent)
  happy = np.zeros(n_agent, dtype=bool)
  prices = np.zeros(n_object)
  
  iter = 0
  while True:
    # Check if all agents are happy
    happy = is_happy(cost_mat=cost_mat,
                     prices=prices,
                     agent_inds=agents,
                     object_inds=assigned_objects,
                     eps=eps)
    if np.all(happy):
      break
    iagent = np.where(~happy)[0][0]

    # If another agent already has the desired object, swap it. Otherwise, assign it directly
    iobject = np.argmax(cost_mat[iagent] - prices)
    if iobject in assigned_objects:
      iswap = np.where(assigned_objects == iobject)[0].item()
      assigned_objects[iswap], assigned_objects[iagent] = \
          assigned_objects[iagent], assigned_objects[iswap]
    else:
      assigned_objects[iagent] = iobject

    # Update the price of the assigned object using equation (8) in Bertsekas1989
    without_best = np.arange(n_object) != iobject
    best = cost_mat[iagent, iobject] - prices[iobject]
    second_best = np.max(cost_mat[iagent][without_best] - prices[without_best])
    prices[iobject] += best - second_best + eps

    iter += 1

  # Return a list of assignment pairs and the computed prices for each object
  if transpose:
    agents, assigned_objects = assigned_objects, agents
  sorted_inds = np.argsort(agents)
  agents = agents[sorted_inds]
  assigned_objects = assigned_objects[sorted_inds]
  pairs = [(a, o) for a, o in zip(agents, assigned_objects)]
  return pairs, prices
