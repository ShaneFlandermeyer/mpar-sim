import uuid
import numpy as np
from mpar_sim.types.state import State
from typing import List

class Trajectory():
  def __init__(self,
               states: List[np.ndarray] = None,
               id: List[np.ndarray] = None):
    self.states = states if states else []
    self.id = id if id else str(uuid.uuid1())

  def step(self, 
           transition_model: callable, 
           nsteps=1, 
           **kwargs):
    for _ in range(nsteps):
      state = transition_model(self.states[-1].state, **kwargs)
      self.append(state, **kwargs)

  def append(self, state: State, **kwargs):
    if isinstance(state, np.ndarray):
      state = State(state, **kwargs)
    self.states.append(state)
    
  def __getitem__(self, index):
    return self.states[index]
  
  def __len__(self):
    return len(self.states)