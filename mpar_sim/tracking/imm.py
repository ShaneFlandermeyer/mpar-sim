from mpar_sim.tracking import KalmanFilter
import numpy as np
from typing import List, Tuple
from scipy.stats import multivariate_normal


class IMMEstimator():
  """
  IMM estimator for tracking a single target with multiple models.
  
  The notation for all mathematical operations follows [1]
  
  Sources
  -------
  [1] Bar-Shalom, Y., Li, X-R., and Kirubarajan, T. “Estimation with Application to Tracking and Navigation”. Wiley-Interscience, 2001.
  """
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
    
    # Step 2: Mix the priors of each filter based on the mixing probabilities computed from step 1
    x_mix, P_mix = self._mixed_init_conds(
        x=states, P=covars, mu_mix=self.mix_probs)
    predicted_states, predicted_covars = [], []
    # Step 3.0: Get predictions from each filter
    for j, filter in enumerate(self.filters):
      x_pred, P_pred = filter.predict(state=x_mix[j],
                                      covar=P_mix[j],
                                      dt=dt)
      predicted_states.append(x_pred)
      predicted_covars.append(P_pred)

    return predicted_states, predicted_covars

  def update(self,
             measurement: np.ndarray,
             predicted_states: List[np.ndarray],
             predicted_covars: List[np.ndarray]) -> Tuple[np.ndarray]:
    r = len(self.filters)
    ndim_state = predicted_states[0].size
    
    # Compute the likelihood 
    likelihoods = np.empty(r)
    x_posts = np.empty((r, ndim_state))
    P_posts = np.empty((r, ndim_state, ndim_state))
    for j in range(r):
      # Step 3.0: Incorporate the measurement into each filter
      x, P, S, _, z_pred = self.filters[j].update(
          measurement=measurement,
          predicted_state=predicted_states[j],
          predicted_covar=predicted_covars[j],
      )
      x_posts[j] = x
      P_posts[j] = P
      # Step 3.5 Compute likelihoods for each mode
      l = multivariate_normal.pdf(
          x=measurement,
          mean=z_pred,
          cov=S,
      )
      likelihoods[j] = l
      

    # Step 4: Update the mode probabilities
    cbar = np.dot(self.mode_probs, self.transition_probs)
    mu = likelihoods * cbar
    self.mode_probs = mu / np.sum(mu)

    # Step 5: Combine model-conditioned state estimates and covariances
    x_mix = np.einsum('j, ji->i', self.mode_probs, x_posts)
    y = x_posts - x_mix
    P_mix = np.einsum('j, jik->ik', self.mode_probs, 
                      P_posts + np.einsum('ji, jk->jik', y, y))
    return x_posts, P_posts, x_mix, P_mix

  # Inidividual algorithm steps as functions from Bar-Shalom2001 section 11.6.6
  @property
  def mix_probs(self):
    # Step 1: compute mixing probabilities
    cbar = np.dot(self.mode_probs, self.transition_probs)
    return np.einsum('ij, i->ij', self.transition_probs, self.mode_probs) / cbar

  def _mixed_init_conds(self,
                        x: List[np.ndarray],
                        P: List[np.ndarray],
                        mu_mix: np.ndarray
                        ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    r = len(x)
    x = np.array(x).reshape(r, -1)
    x_mix = np.einsum('ij, ik->jk', mu_mix, x)

    P_mix = np.zeros_like(P)
    for j in range(r):
      y = x - x_mix[j]
      P_mix[j] = np.einsum('j, jik->ik', mu_mix[:, j],
                            P + np.einsum('ji, jk->jik', y, y))
    return list(x_mix), list(P_mix)