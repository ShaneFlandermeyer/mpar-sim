from mpar_sim.tracking import KalmanFilter
import numpy as np
from typing import List, Tuple
from scipy.stats import multivariate_normal


class IMMEstimator():
  def __init__(self,
               filters: List[KalmanFilter],
               init_probs: np.ndarray,
               transition_probs: np.ndarray,
               ):
    self.filters = filters
    self.mode_probs = init_probs
    self.transition_probs = transition_probs

  def predict(self,
              states: List[np.ndarray],
              covars: List[np.ndarray],
              dt: float) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    # Step 1: Update mixing probabilities
    mix_probs = self._mixing_probs(self.mode_probs, self.transition_probs)

    # Step 2: Mix initial conditions
    mixed_states, mixed_covars = self._mixed_init_conds(
        x=states, P=covars, mu=mix_probs)
    predicted_states, predicted_covars = [], []
    for i, filter in enumerate(self.filters):
      x_pred, P_pred = filter.predict(state=mixed_states[i],
                                      covar=mixed_covars[i],
                                      dt=dt)
      predicted_states.append(x_pred)
      predicted_covars.append(P_pred)

    return predicted_states, predicted_covars

  def update(self,
             measurement: np.ndarray,
             predicted_states: List[np.ndarray],
             predicted_covars: List[np.ndarray]) -> Tuple[np.ndarray]:
    likelihoods = []
    for i, f in enumerate(self.filters):
      _, _, S, _, z_pred = f.update(
          measurement=measurement,
          predicted_state=predicted_states[i],
          predicted_covar=predicted_covars[i],
      )
      l = multivariate_normal.pdf(
        x=measurement,
        mean=z_pred,
        cov=S,
      )
      likelihoods.append(l)
    # TODO: Update the mode probability
    # TODO: Estimate and covariance combination
    pass

  # Inidividual algorithm steps as functions from Bar-Shalom2001 section 11.6.6
  def _mixing_probs(self, mu: np.ndarray, M: np.ndarray):
    cbar = np.dot(mu, M)
    return np.einsum('ij, i->ij', M, mu) / cbar

  def _mixed_init_conds(self,
                        x: List[np.ndarray],
                        P: List[np.ndarray],
                        mu: np.ndarray
                        ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    r = len(x)
    x = np.array(x).reshape(r, -1)
    x_mix = np.einsum('ij, ik->jk', mu, x)

    # TODO: SPEED THIS UP or numbafy it
    P = np.array(P)
    P_mix = np.zeros_like(P)
    for j in range(r):
      for i in range(r):
        y = x[i] - x_mix[j]
        P_mix[j] += mu[i, j] * (P[i] + np.outer(y, y))

    return list(x_mix), list(P_mix)

  def _mode_matched_filter(self):
    # Step 3
    pass

  def _update_mode_probabilities(self):
    # Step 4
    pass

  def _mix_estimates(self):
    # Step 5
    pass
