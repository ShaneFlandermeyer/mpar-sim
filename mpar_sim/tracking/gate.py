import numpy as np
from typing import List
from scipy.stats import norm
import math

def ellipsoid_gate(
  measurements: List[np.ndarray], 
  predicted_measurement: np.ndarray, 
  innovation_covar: np.ndarray, 
  threshold: float
  ) -> List[bool]:
  """
  Determine which measurements fall within the ellipsoidal gate centered around a predicted measurement.

  Parameters
  ----------
  measurements : List[np.ndarray]
      _description_
  predicted_measurement : np.ndarray
      _description_
  innovation_covar : np.ndarray
      _description_
  threshold : float
      _description_

  Returns
  -------
  List[bool]
      _description_
  """
  z = np.array([m.reshape((-1, 1)) for m in measurements])
  z_pred = predicted_measurement.reshape((-1, 1))
  y = z - z_pred
  Si = np.linalg.inv(innovation_covar)
  
  # Compute the distance for all measurements
  dist = np.einsum('ii, nij -> nij', Si, y)
  dist = np.einsum('nij, nji -> n', y.swapaxes(-1, -2), dist)
  return dist <= threshold

def gate_probability(gate_dim: int, threshold: float) -> float:
  """
  Compute the probability of a measurement falling within the ellipsoidal gate.

  Parameters
  ----------
  gate_dim : int
      _description_
  threshold : float
      _description_

  Returns
  -------
  float
      _description_
  """
  # Standard Gaussian probability integral
  gc = lambda G : norm.cdf(G) - norm.cdf(0)
  G = threshold
  sqrt_G = np.sqrt(G)
  if gate_dim == 1:
    return 2*gc(sqrt_G)
  elif gate_dim == 2:
    return 1 - np.exp(-G/2)
  elif gate_dim == 3:
    return 2*gc(sqrt_G) - np.sqrt(2*G/np.pi)*np.exp(-G/2)
  elif gate_dim == 4:
    return 1 - (1+G/2)*np.exp(-G/2)
  elif gate_dim == 5:
    return 2*gc(sqrt_G) - (1+G/3)*np.sqrt(2*G/np.pi)*np.exp(-G/2)
  elif gate_dim == 6:
    return 1 - 0.5*(G**2/4+G+2)*np.exp(-G/2)

if __name__ == '__main__':
  measurements = [np.array([1, 1]), np.array([2, 2]), np.array([3, 3])]
  predicted_measurement = np.array([1, 1])
  innovation_covar = np.eye(2)
  threshold = 7
  print(ellipsoid_gate(measurements, predicted_measurement, innovation_covar, threshold))
  print(gate_probability(2, threshold))