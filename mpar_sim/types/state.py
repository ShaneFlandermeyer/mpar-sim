import datetime
from typing import Union
import numpy as np

class State():
  def __init__(
      self,
      state: np.ndarray,
      timestamp: Union[float, datetime.datetime] = None,
      metadata: dict = {},
  ):
    self.state = state
    self.timestamp = timestamp
    self.metadata = metadata
