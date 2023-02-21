import datetime
from typing import Callable, Tuple, Union

import numpy as np
from mpar_sim.types.detection import Detection
from mpar_sim.types.state import State

from mpar_sim.types.track import Track


class Tracker():
  def __init__(self,
               predict_func: Callable,
               update_func: Callable,
               transition_model: Callable,
               measurement_model: Callable,
               prior_state: State = None,
               ):
    self.predict_func = predict_func
    self.update_func = update_func
    self.transition_model = transition_model
    self.measurement_model = measurement_model
    self.prior_state = prior_state

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

    TODO: Assumes the measurement model is reversible and nonlinear.
    TODO: Add a prior_state field to the object to initialize state parameters that cannot be estimated from the measurement. Assuming zero for now
    # TODO: Add diagonal loading to covariance for numerical stability
    # TODO: Assumes a TrueDetection (i.e., measurement.groundtruth_path exists)


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
    model_matrix = self.measurement_model.jacobian(state_vector)
    inv_model_matrix = np.linalg.pinv(model_matrix)
    model_covar = self.measurement_model.covar()
    covar = inv_model_matrix @ model_covar @ inv_model_matrix.T
    state = State(state_vector, covar, measurement.timestamp)

    # If a prior state is provided, use it to initialize the parts of the state that cannot be estimated from the measurement
    if self.prior_state is not None:
      prior_state_vector = self.prior_state.state_vector.copy()
      prior_covar = self.prior_state.covar.copy()

      mapped_dimensions = self.measurement_model.position_mapping
      prior_state_vector[mapped_dimensions] = 0
      prior_covar[mapped_dimensions] = 0

      state_vector += prior_state_vector.reshape(state_vector.shape)
      covar += prior_covar.reshape(covar.shape)

    return Track(state, target_id=measurement.groundtruth_path.id)
