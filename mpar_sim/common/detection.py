#
# Author: Shane Flandermeyer
# Created on Fri Jun 30 2023
# Copyright (c) 2023
#
# Currently has Swerling model stuff. See Mahafza 2013 for the math.
#


from typing import Collection
import jax.numpy as jnp
from jax.scipy.special import erfc, gammainc

def logfactorial(n):
  """
  Compute the log factorial of n
  
  See: https://math.stackexchange.com/questions/138194/approximating-log-of-factorial
  """
  m = n*(1 + 4*n*(1 + 2*n))
  return n*(jnp.log(n) - 1) + (1/2)*(1/3*jnp.log(1/30 + m) + jnp.log(jnp.pi))

def threshold(nfa: Collection[float], n_pulse: Collection[int]):
  """
  This function calculates the threshold value from nfa and np
  using the newton-Raphson recursive formula

  Parameters
  ----------
  nfa : float
      Number of false alarms
  npulse : int
      Number of pulses
  """
  eps = jnp.finfo(float).eps
  pfa = n_pulse * jnp.log(2) / nfa
  sqrt_pfa = jnp.sqrt(-jnp.log10(pfa))
  vt0 = n_pulse - jnp.sqrt(n_pulse) + 2.3 * sqrt_pfa * (sqrt_pfa + jnp.sqrt(n_pulse) - 1.0)
  vt = vt0
  while True:
    igf = gammainc(n_pulse, vt0)
    num = 0.5**(n_pulse/nfa) - igf
    deno = -jnp.exp(-vt0 + jnp.log(vt0)*(n_pulse-1) - logfactorial(n_pulse-1))
    vt = vt0 - (num / (deno + eps))
    delta = abs(vt - vt0)
    vt0 = vt
    if jnp.all(delta < 1e-4*vt0):
      break
  return vt

def pd_swerling0(pfa: float, n_pulse: float, snr_db: float) -> float:
  """
  Compute the probability of detection for a swerling 0/5 target for non-coherently integrated pulses.
  

  Parameters
  ----------
  pfa : float
      Probability of false alarm
  n_pulse : float
      Number of pulses
  snr_db : float
      Single-pulse SNR in dB
  """
  
  nfa = n_pulse * jnp.log(2) / pfa
  vt = threshold(nfa, n_pulse)
  
  # Compute the Gram-Charlier coefficients
  snr = 10**(snr_db/10)
  C3 = -(snr + 1/3) / (jnp.sqrt(n_pulse)*(2*snr + 1)**1.5)
  C4 = (snr + 1/4) / (n_pulse*(2*snr + 1)**2)
  C6 = C3**2 / 2
  w = jnp.sqrt(n_pulse*(2*snr + 1))
  
  V = (vt - n_pulse*(1 + snr)) / w
  pd = 0.5*erfc(V / jnp.sqrt(2)) - \
    (jnp.exp(-V**2/2) / jnp.sqrt(2*jnp.pi)) * \
    (C3*(V**2 - 1) + C4*V*(3-V**2) - C6*V*(V**4 - 10*V**2 + 15))
    
  return pd

def pd_swerling1(pfa: float, n_pulse: float, snr_db: float) -> float:
  """
  Compute the probability of detection for a swerling 0/5 target for non-coherently integrated pulses.
  

  Parameters
  ----------
  pfa : float
      Probability of false alarm
  n_pulse : float
      Number of pulses
  snr_db : float
      Single-pulse SNR in dB
  """
  
  nfa = n_pulse * jnp.log(2) / pfa
  vt = threshold(nfa, n_pulse)
  snr = 10**(snr_db/10)
  
  if n_pulse == 1:
    pd = jnp.exp(-vt / (1 + snr))
  else:
    pd = 1 - gammainc(n_pulse-1, vt) + \
      (1 + 1/(n_pulse*snr))**(n_pulse - 1) * \
      gammainc(n_pulse-1, vt / (1 + 1/(n_pulse*snr))) * \
      jnp.exp(-vt / (1 + n_pulse*snr))
    
  return pd

def pd_swerling2(pfa: float, n_pulse: float, snr_db: float):
  # TODO:
  pass

def pd_swerling3(pfa, n_pulse, snr_db):
  # TODO: 
  pass

def pd_swerling4(pfa, n_pulse, snr_db):
  # TODO: 
  pass

if __name__ == '__main__':
  pfa = jnp.array([1e-6]*2)
  n_pulse = jnp.array([5]*2)
  snr_db = jnp.array([5]*2)
  pd_swerling0(pfa, n_pulse, snr_db)
  