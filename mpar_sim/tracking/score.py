import numpy as np
from mpar_sim.types import Track
from typing import List


class TrackScoreManager():
  def __init__(self,
               beta: float = 0.1,
               volume: float = 1,
               pd: float = 0.9,
               pfa: float = 1e-6,
               ):
    self.beta = beta
    self.volume = volume
    self.pd = pd
    self.pfa = pfa

  def update(self,
             measurement: np.ndarray,
             track: np.ndarray,
             likelihood: float = None,
             ) -> float:
    """
    Update the score of the input track from a measurement (or no measurement)

    Parameters
    ----------
    measurement : np.ndarray
        _description_
    track : np.ndarray
        _description_
    likelihood : float, optional
        Likelihood that the measurement is associated to the track, by default None.
        If the residual is Gaussian and the measurement is M-dimensional, the likelihood function is:
        g = exp(-d^2/2) / ((2*pi)^(M/2) * sqrt(det(S)))
    innovation_covar : np.ndarray, optional
        _description_, by default None
    d : float, optional
        _description_, by default None

    Returns
    -------
    float
        Updated track score
    """
    # Initialize track score if no score exists for the track
    if not hasattr(track, 'score'):
      track.score = np.log(self.beta*self.volume) + np.log(self.pd/self.pfa)
      return track.score

    # No measurement for the track
    if measurement is None:
      track.score += np.log(1 - self.pd)
      return track.score

    # Update score based on (18) in Werthmann1992
    track.score += np.log(likelihood * self.volume)
    track.score += np.log(self.pd/self.pfa)
    return track.score


if __name__ == '__main__':
  m = TrackScoreManager(beta=0.1, volume=1.3, pd=0.9, pfa=1e-6)
  t = Track()
  m.update(measurement=None, track=t)
  # print(t.score)

  # m.update(measurement=None, track=t)
  # m.update(measurement=np.zeros(6),
  #          track=t,
  #          likelihood=0.05 + 0.05)
  for i in range(10):
    s = m.update(measurement=np.zeros(6),
                 track=t,
                 likelihood=0.05 + 0.025)
    # m.update(measurement=None, track=t)
    print(t.score)
