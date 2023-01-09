import datetime
from typing import List
import uuid
import numpy as np


class GroundTruthState():
  """
  Stores information about the current state of the ground truth.
  """
  def __init__(self,
               state_vector: np.ndarray,
               timestamp: datetime.datetime = None,
               metadata: dict = {},
               ):
    if state_vector.ndim == 1:
      state_vector = state_vector.reshape(-1, 1)
    self.state_vector = state_vector
    self.timestamp = timestamp
    self.metadata = metadata


class GroundTruthPath():
  """
  Stores the entire history of the ground truth target.
  """
  def __init__(self,
               states: List[GroundTruthState] = [],
               id=None):
    self.states = states
    if id is None:
      self.id = str(uuid.uuid1())
    else:
      self.id = id

  def append(self, state: GroundTruthState):
    self.states.append(state)
    
  def __getitem__(self, index):
    return self.states[index]

  def __len__(self):
    return len(self.states)