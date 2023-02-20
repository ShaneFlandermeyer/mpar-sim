import datetime
from typing import Callable, Tuple, Union

import numpy as np

from mpar_sim.types.state import State
from mpar_sim.types.track import Track


class TestLinearTransitionModel():
  def function(self, prior, dt, noise=False):
    if noise:
      process_noise = self.covar()
    else:
      process_noise = 0

    return self.matrix(dt) @ prior + process_noise

  def matrix(self, dt):
    return np.array([[1, dt], [0, 1]])

  def covar(self, dt=None):
    return np.array([[0.588, 1.175],
                     [1.175, 2.35]])


class Tracker():
  def __init__(self,
               predict_func: Callable,
               update_func: Callable,
               transition_model: Callable,
               measurement_model: Callable,
               #  associate_func: Callable,
               ):
    self.predict_func = predict_func
    self.update_func = update_func
    self.transition_model = transition_model
    self.measurement_model = measurement_model
    # self.associate_func = associate_func

  def predict(self,
              track: Track,
              time: Union[float, datetime.datetime]
              ) -> Tuple[np.ndarray, np.ndarray]:

    # Ensure the time and track timestamp are the same type (datetime.datetime or float)
    assert isinstance(time, type(track.timestamp))

    return self.predict_func(state=track.state_vector,
                             covar=track.covar,
                             transition_model=self.transition_model,
                             time_interval=time - track.timestamp)

  def update(self,
             track: Track,
             measurement: np.ndarray,
             ) -> Tuple[np.ndarray, np.ndarray]:
    return self.update_func(state=track.state_vector,
                            covar=track.covar,
                            measurement=measurement,
                            measurement_model=self.measurement_model)