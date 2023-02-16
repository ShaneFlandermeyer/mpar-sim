import uuid
from mpar_sim.types.state import State
from typing import List


class Track():
  """
  Helper class for storing the state histories of a track. 
  
  This class is functionally identical to the GroundTruthPath class
  """
  def __init__(self,
               states: List[State] = [],
               id=None):
    self.states = states
    if id is None:
      self.id = str(uuid.uuid1())
    else:
      self.id = id

  def append(self, state: State):
    self.states.append(state)

  @property
  def state_vector(self):
    return self.states[-1].state_vector

  @property
  def timestamp(self):
    return self.states[-1].timestamp

  @property
  def metadata(self):
    return self.states[-1].metadata

  def __getitem__(self, index):
    return self.states[index]

  def __len__(self):
    return len(self.states)
