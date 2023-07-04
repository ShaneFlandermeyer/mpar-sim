from typing import Union
import jax
import numpy as np
from mpar_sim.models.transition.base import TransitionModel
from mpar_sim.models.rcs import RCSModel, Swerling
import random


class Target():
  def __init__(
      self,
      position: np.array = None,
      velocity: np.array = None,
      transition_model: TransitionModel = None,
      rcs: Union[RCSModel, float] = None,
      seed: int = random.randint(0, 2**32-1)
  ) -> None:
    self.position = np.array(position)
    self.velocity = np.array(velocity)
    self.transition_model = transition_model
    self.rcs_model = rcs
    if isinstance(self.rcs_model, float):
      self.rcs_model = Swerling(case=0, mean=self.rcs_model, seed=seed)

  def move(self, **kwargs) -> None:
    # Collect state vector
    pos_inds = self.transition_model.position_mapping
    vel_inds = self.transition_model.velocity_mapping
    state = np.zeros(self.transition_model.ndim_state)
    state[pos_inds] = self.position.ravel()
    state[vel_inds] = self.velocity.ravel()
    # Pass to transition model and update the target state
    state = self.transition_model(state=state, **kwargs)
    self.position = state[self.transition_model.position_mapping].reshape(
        self.position.shape)
    self.velocity = state[self.transition_model.velocity_mapping].reshape(
        self.velocity.shape)

  @property
  def rcs(self, **kwargs) -> np.array:
    return self.rcs_model(**kwargs)

  def detection_probability(self, pfa, n_pulse, snr_db) -> float:
    return self.rcs_model.detection_probability(pfa=pfa, n_pulse=n_pulse, snr_db=snr_db)
