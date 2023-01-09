import datetime
from typing import Tuple

import numpy as np

from mpar_sim.models.transition.base import TransitionModel


def kalman_predict(prior_state: np.ndarray,
                   prior_covar: np.ndarray,
                   transition_matrix: np.ndarray,
                   noise_covar: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
  """
  Kalman predict step.

  Equations: https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python/blob/master/06-Multivariate-Kalman-Filters.ipynb

  Parameters
  ----------
  prior_state : np.ndarray
      Prior state vector
  prior_covar : np.ndarray
      Prior covariance matrix
  transition_matrix : np.ndarray
      Transition model matrix
  noise_covar : np.ndarray
  Returns
  -------
  Tuple[np.ndarray, np.ndarray]
      - Predicted state vector
      - Predicted covariance matrix
  """

  if np.isscalar(transition_matrix):
      transition_matrix = np.array(transition_matrix)
      
  predicted_state = transition_matrix @ prior_state
  predicted_covar = transition_matrix @ prior_covar @ transition_matrix.T + noise_covar


  return predicted_state, predicted_covar
