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
               filter=None,
               ) -> None:
    if isinstance(history, np.ndarray):
      history = [State(history)]
    elif isinstance(history, State):
      history = [history]
    self.history = history if history is not None else []
    self.id = id if id else str(uuid.uuid1())
    self.filter = filter

  def append(self, state: State, **kwargs):
    if isinstance(state, np.ndarray):
      state = State(state, **kwargs)
    self.history.append(state)

  def __getitem__(self, index):
    return self.history[index]

  def __len__(self):
    return len(self.history)

  def predict(self, dt: float, **kwargs):
    return self.filter.predict(dt=dt,
                               state=self.state,
                               covar=self.covar,
                               **kwargs)

  def update(self, **kwargs):
    return self.filter.update(**kwargs)

  @property
  def state(self):
    return self.history[-1].state

  @property
  def covar(self):
    return self.history[-1].covar
