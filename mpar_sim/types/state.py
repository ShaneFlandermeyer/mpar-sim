import datetime
import numpy as np


class GaussianState():
  def __init__(self,
               state_vector: np.ndarray,
               covar: np.ndarray,
               timestamp: datetime.datetime = None):
    """
    A class representing a Gaussian state, which is characterized by a mean (the state vector) and covariance matrix. The covariance matrix must be square and must have the same dimension as the state vector.

    Parameters
    ----------
    state_vector : np.ndarray
        The state vector, which is the mean of the Gaussian state.
    covar : np.ndarray
        The covariance matrix.

    Raises
    ------
    ValueError
        _description_
    """
    self.state_vector = state_vector
    self.covar = covar
    if self.state_vector.shape[0] != self.covar.shape[0]:
      raise ValueError(
          'State vector and covariance matrix must have the same dimension')
    self.timestamp = timestamp

  @property
  def mean(self):
    return self.state_vector

  @property
  def ndim(self):
    return self.state_vector.shape[0]
