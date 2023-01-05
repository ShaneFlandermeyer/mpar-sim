import datetime
from typing import Tuple

import numpy as np

from mpar_sim.models.transition.base import TransitionModel


def kalman_predict(prior_state: np.ndarray,
                   prior_covar: np.ndarray,
                   transition_model: TransitionModel,
                   timestamp: datetime.datetime) -> Tuple[np.ndarray, np.ndarray]:
  """
  Kalman predict step.

  Equations: https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python/blob/master/06-Multivariate-Kalman-Filters.ipynb

  Parameters
  ----------
  prior_state : np.ndarray
      Prior state vector
  prior_covar : np.ndarray
      Prior covariance matrix
  transition_model : TransitionModel
      Transition model
  timestamp : datetime.datetime
      Predict the state to this time

  Returns
  -------
  Tuple[np.ndarray, np.ndarray]
      - Predicted state vector
      - Predicted covariance matrix
  """
  if timestamp is None or prior_state.timestamp is None:
    dt = None
  else:
    dt = timestamp - prior_state.timestamp

  # Predict the mean
  predicted_state = transition_model.function(prior_state, time_interval=dt, noise=True)
  # Predict the covariance
  transition_matrix = transition_model.matrix(time_interval=dt)
  
  process_noise = transition_model.covar(time_interval=dt)
  predicted_covar = transition_matrix @ prior_covar @ transition_matrix.T + process_noise
  
  return predicted_state, predicted_covar
