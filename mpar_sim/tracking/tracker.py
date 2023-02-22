import datetime
from typing import Callable, Tuple, Union

import numpy as np
from mpar_sim.common.coordinate_transform import cart2sph_covar, sph2cart_covar
from mpar_sim.types.detection import Detection, TrueDetection
from mpar_sim.types.state import State

from mpar_sim.types.track import Track


class Tracker():
  def __init__(self,
               predict_func: Callable,
               update_func: Callable,
               transition_model: Callable,
               measurement_model: Callable,
               ):
    self.predict_func = predict_func
    self.update_func = update_func
    self.transition_model = transition_model
    self.measurement_model = measurement_model

  def predict(self,
              state: State,
              time: Union[float, datetime.datetime]
              ) -> Tuple[np.ndarray, np.ndarray]:

    # Ensure the time and track timestamp are the same type (datetime.datetime or float)
    last_update_time = state.timestamp
    if isinstance(last_update_time, int):
      last_update_time = float(state.timestamp)
    assert isinstance(time, type(last_update_time))

    return self.predict_func(state=state.state_vector,
                             covar=state.covar,
                             transition_model=self.transition_model,
                             time_interval=time - state.timestamp)

  def update(self,
             state: State,
             measurement: np.ndarray,
             ) -> Tuple[np.ndarray, np.ndarray]:

    return self.update_func(state=state.state_vector,
                            covar=state.covar,
                            measurement=measurement,
                            measurement_model=self.measurement_model)

  def initiate(self, measurement: Detection) -> Track:
    """
    Initiate a new track from a measurement.

    Assumes the measurement model is reversible and nonlinear.
    # TODO: Add diagonal loading to covariance for numerical stability


    Parameters
    ----------
    measurement : np.ndarray
        The input measurement.

    Returns
    -------
    Track
        Output track.
    """
    # Use the measurement model to estimate the state vector and covariance from the measurement
    state_vector = self.measurement_model.inverse_function(
        measurement.state_vector)
    model_covar = self.measurement_model.covar()
    pos_covar, vel_covar = sph2cart_covar(
        model_covar, *measurement.state_vector.ravel())

    pos_mapping = self.measurement_model.position_mapping
    vel_mapping = self.measurement_model.velocity_mapping
    covar = np.zeros((6, 6))
    covar[np.ix_(pos_mapping, pos_mapping)] = pos_covar
    covar[np.ix_(vel_mapping, vel_mapping)] = vel_covar

    # If the measurement is a TrueDetection (corresponds to an actual target), store the target ID in the track object
    if isinstance(measurement, TrueDetection):
      target_id = measurement.groundtruth_path.id
    else:
      target_id = None

    state = State(state_vector, covar, measurement.timestamp)
    return Track(state, target_id=target_id)
