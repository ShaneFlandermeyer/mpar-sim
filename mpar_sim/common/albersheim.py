import numpy as np
from typing import Union

def albersheim_pd(
        snr: Union[np.ndarray, float],
        false_alarm_probability: Union[np.ndarray, float],
        N: int) -> float:
  """
  Compute the probability of detection for a given SNR and false alarm rate using Albersheim's equation.

  Albersheim's equation assumes the target is nonfluctuating, integration is non-coherent, and a linear detector is used. 

  See: http://www.radarsp.com/

  Parameters
  ----------
  snr : float
      Signal-to-noise ratio
  false_alarm_probability : float
      False alarm probability
  N : int
      Number of pulses to noncoherently integrate

  Returns
  -------
  float
      Probability of detection
  """
  A = np.log(0.62 / false_alarm_probability)
  Z = (snr + 5*np.log10(N)) / (6.2 + (4.54 / np.sqrt(N + 0.44)))
  B = (10**Z - A) / (1.7 + 0.12*A)
  Pd = 1 / (1 + np.exp(-B))
  return Pd
