import uuid
from mpar_sim.types.state import State
from typing import List, Union, Optional
import numpy as np


class Track():
  """
  Helper class for storing the state history of a track. 

  """

  def __init__(self,
               history: Union[State, List[State]] = None,
               id: Union[str, int] = None,
               ) -> None:
    self.history = history if history else []
    self.id = id if id else str(uuid.uuid1())

  def append(self, state: State, **kwargs):
    if isinstance(state, np.ndarray):
      state = State(state, **kwargs)
    self.history.append(state)
    
  def __getitem__(self, index):
    return self.history[index]

  def __len__(self):
    return len(self.history)
  
  @property
  def state(self):
    """Returns the most recent state vector"""
    return self.history[-1].state if self.history else None

  @property
  def covar(self):
    """Returns the most recent covariance matrix"""
    return self.history[-1].covar
  
