import datetime
from typing import Tuple
import numpy as np


class KalmanPredictor():
  def __init__(self,
               transition_model,
               ) -> None:
    self.transition_model = transition_model

  def predict(self,
              prior: np.ndarray,
              timestamp: datetime.datetime = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return the predicted state vector and covariance for the given prior at the given time.

    Parameters
    ----------
    prior : _type_
        _description_
    timestamp : _type_, optional
        _description_, by default None

    Returns
    -------
    _type_
        _description_
    """
    # Compute the time interval between the prior and the timestamp
    if timestamp is None or prior.timestamp is None:
      dt = None
    else:
      dt = timestamp - prior.timestamp

    # Predict the mean
    x = self.transition_model.function(prior, time_interval=dt)

    # Predict the covariance
    P_prior = prior.covar
    F = self.transition_model.matrix(time_interval=dt)
    Q = self.transition_model.covar(time_interval=dt)
    P = F @ P_prior @ F.T + Q

    return x, P
