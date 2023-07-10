import datetime
from typing import List, Union
import uuid
import numpy as np
from mpar_sim.types.state import State


class Trajectory():
  def __init__(self,
               states = [],
               transition_model = None,
               id = None):
    if not isinstance(states, list):
      states = [states]
    self.states = states
    self.id = id if id else str(uuid.uuid1())

  def step(self, 
           transition_model: callable, 
           nsteps=1, 
           **kwargs):
    for _ in range(nsteps):
      state = transition_model(self.states[-1].state, **kwargs)
      self.append(state)

  def append(self, state: State, **kwargs):
    if isinstance(state, np.ndarray):
      state = State(state, **kwargs)
    self.states.append(state)
    
  def __getitem__(self, index):
    return self.states[index]
  
  def __len__(self):
    return len(self.states)