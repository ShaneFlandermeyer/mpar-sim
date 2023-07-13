import uuid
from mpar_sim.types.state import State
from typing import List, Union, Optional


class Track():
  """
  Helper class for storing the state history of a track. 

  """

  def __init__(self,
               history: Union[State, List[State]] = None,
               id: Union[str, int] = None,
               target_id: Optional[Union[str, int]] = None
               ) -> None:
    self.history = history if history else []
    if id is None:
      self.id = str(uuid.uuid1())
    else:
      self.id = id

    # NOTE: This is a workaround to avoid needing a data association algorithm. You can just "associate" a detection by checking for a matching target ID.
    self.target_id = target_id

  def append(self, state: State):
    self.history.append(state)

  @property
  def state_vector(self):
    """Returns the most recent state vector"""
    return self.history[-1].state_vector

  @property
  def covar(self):
    """Returns the most recent covariance matrix"""
    return self.history[-1].covar

  @property
  def timestamp(self):
    """Returns the most recent track update time"""
    return self.history[-1].timestamp

  @property
  def metadata(self):
    """Returns metadata for the most recent track update"""
    return self.history[-1].metadata

  def __getitem__(self, index):
    return self.history[index]

  def __len__(self):
    return len(self.history)
