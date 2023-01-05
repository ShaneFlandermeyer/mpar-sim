import datetime
from typing import Tuple

import numpy as np

from mpar_sim.models.transition.base import TransitionModel


def kalman_predict(prior: np.ndarray,
                   transition_model: TransitionModel,
                   timestamp: datetime.datetime) -> Tuple[np.ndarray, np.ndarray]:
  """
  Kalman predict step.
  
  Equations: https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python/blob/master/06-Multivariate-Kalman-Filters.ipynb

  Parameters
  ----------
  prior : np.ndarray
      Prior state vector
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
  if timestamp is None or prior.timestamp is None:
    dt = None
  else:
    dt = timestamp - prior.timestamp

  # Predict the mean
  x = transition_model.function(prior, time_interval=dt)
  # Predict the covariance
  P_prior = prior.covar
  F = transition_model.matrix(time_interval=dt)
  Q = transition_model.covar(time_interval=dt)
  P = F @ P_prior @ F.T + Q
  return x, P
