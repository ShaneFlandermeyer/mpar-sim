import numpy as np

def range_crlb(snr: float, resolution: float, bias_fraction: float = 0) -> float:
  """
  Computes the Cramer-Rao lower bound on the range estimation accuracy.

  Parameters
  ----------
  snr : float
      Single-pulse SNR in linear units.
  resolution : float
      RMS range resolution.

  Returns
  -------
  float
      Cramer-Rao lower bound variance.
  """
  # Richards 2014 - Eq. (7.36)
  improvement_factor = 1 / (8*np.pi**2*snr)
  improvement_factor += bias_fraction**2
  # Limit worst-case accuracy to uniformly distributed over resolution
  improvement_factor = np.clip(improvement_factor, 0, 1/12)
  variance = resolution**2 * improvement_factor

  return variance

def velocity_crlb(snr: float, resolution: float, bias_fraction: float = 0):
  """
  Computes the Cramer-Rao lower bound on the velocty estimation accuracy.

  Parameters
  ----------
  snr : float
      SNR (after coherent integration) in linear units.
  resolution : float
      velocity resolution.
  """
  # Richard 2014 - Eq. (7.64)
  improvement_factor = 6 / ((2*np.pi)**2 * snr)
  improvement_factor += bias_fraction**2
  # Limit worst-case accuracy to uniformly distributed over resolution
  improvement_factor = np.clip(improvement_factor, 0, 1/12)
  variance = resolution**2 * improvement_factor
  return variance

def angle_crlb(snr: float, resolution: float, bias_fraction: float = 0):
  """
  Compute the Cramer-Rao lower bound on the angle estimation accuracy

  Parameters
  ----------
  snr : float
      Coherently integrated SNR (across the array elements) in linear units.   
  resolution : float
      Angular resolution in degrees
  """

  resolution = np.deg2rad(resolution)

  # Skolnik2002 - Eq. (6.37)
  # This value of k assumes uniform weighting across the aperture for sidelobe control
  k = 0.886
  improvement_factor = 6 / ((2*np.pi)**2 * snr * k**2)
  improvement_factor += bias_fraction**2
  # Limit worst-case accuracy to uniformly distributed over resolution
  improvement_factor = np.clip(improvement_factor, 0, 1/12)
  variance = resolution**2 * improvement_factor
  return variance
