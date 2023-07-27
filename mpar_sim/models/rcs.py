import numpy as np
from scipy.special import erfc, gammainc


class RCSModel():
  """Base class for RCS models"""


class Swerling(RCSModel):
  def __init__(self,
               case: int = 0,
               mean: float = 1,
               seed: int = np.random.randint(0, 2**32-1)):
    self.case = case
    self.mean = mean
    self.np_random = np.random.RandomState(seed)

  def __call__(self):
    """
    Sample an RCS value from the model by transforming a Uniform(0, 1) random variable.
    
    See: https://radarsp.weebly.com/uploads/2/1/4/7/21471216/generating_swerling_random_sequences.pdf

    Returns
    -------
    float
        Target RCS at the current time
    """
    if self.case in [0, 5]:
      return self.mean
    elif self.case in [1, 2]:
      u = self.np_random.uniform()
      x = -self.mean * np.log(u)
      return x
    elif self.case in [3, 4]:
      u = self.np_random.uniform(size=2)
      x = -self.mean/2 * np.log(u[0]*u[1])
      return x
    else:
      raise NotImplementedError

  def detection_probability(self, pfa, n_pulse, snr_db):
    if self.case == 0:
      return pd_swerling0(pfa, n_pulse, snr_db)
    elif self.case == 1:
      return pd_swerling1(pfa, n_pulse, snr_db)
    elif self.case == 2:
      return pd_swerling2(pfa, n_pulse, snr_db)
    elif self.case == 3:
      return pd_swerling3(pfa, n_pulse, snr_db)
    else:
      raise NotImplementedError

def logfactorial(n):
  """
  Compute the log factorial of n

  See: https://math.stackexchange.com/questions/138194/approximating-log-of-factorial
  """
  m = n*(1 + 4*n*(1 + 2*n))
  return n*(np.log(n) - 1) + (1/2)*(1/3*np.log(1/30 + m) + np.log(np.pi))

def threshold(nfa: float, n_pulse: int, niter: int = 1):
  """
  This function calculates the threshold value from nfa and np
  using the newton-Raphson recursive formula

  Parameters
  ----------
  nfa : float
      Number of false alarms
  npulse : int
      Number of pulses
  niter : int
      Number of iterations
  """
  eps = np.finfo(float).eps
  pfa = n_pulse * np.log(2) / nfa
  sqrt_pfa = np.sqrt(-np.log10(pfa))
  vt0 = n_pulse - np.sqrt(n_pulse) + 2.3 * sqrt_pfa * \
      (sqrt_pfa + np.sqrt(n_pulse) - 1.0)
  vt = vt0
  for _ in range(niter):
    igf = gammainc(n_pulse, vt0)
    num = 0.5**(n_pulse/nfa) - igf
    den = -np.exp(-vt0 + np.log(vt0)*(n_pulse-1) - 
                   logfactorial(n_pulse-1+eps))
    vt = vt0 - (num / (den + eps))
    vt0 = vt
  return vt

def pd_swerling0(pfa: float, n_pulse: float, snr_db: float) -> float:
  """
  Compute the probability of detection for a swerling 0/5 target for non-coherently integrated pulses.

  See: Mahafza 2013

  Parameters
  ----------
  pfa : float
      Probability of false alarm
  n_pulse : float
      Number of pulses
  snr_db : float
      Single-pulse SNR in dB
  """
  snr = 10**(snr_db/10)
  if n_pulse == 1:
    return 0.5 * erfc(np.sqrt(-np.log(pfa)) - np.sqrt(snr + 0.5))

  nfa = n_pulse * np.log(2) / pfa
  vt = threshold(nfa, n_pulse)

  # Compute the Gram-Charlier coefficients
  C3 = -(snr + 1/3) / (np.sqrt(n_pulse)*(2*snr + 1)**1.5)
  C4 = (snr + 1/4) / (n_pulse*(2*snr + 1)**2)
  C6 = C3**2 / 2
  w = np.sqrt(n_pulse*(2*snr + 1))

  V = (vt - n_pulse*(1 + snr)) / w
  pd = 0.5*erfc(V / np.sqrt(2)) - \
      (np.exp(-V**2/2) / np.sqrt(2*np.pi)) * \
      (C3*(V**2 - 1) + C4*V*(3-V**2) - C6*V*(V**4 - 10*V**2 + 15))

  return pd

def pd_swerling1(pfa: float, n_pulse: float, snr_db: float) -> float:
  """
  Compute the probability of detection for a swerling 1 target for non-coherently integrated pulses.

  See: Mahafza 2013

  Parameters
  ----------
  pfa : float
      Probability of false alarm
  n_pulse : float
      Number of pulses
  snr_db : float
      Single-pulse SNR in dB
  """
  snr = 10**(snr_db/10)
  nfa = n_pulse * np.log(2) / pfa
  vt = threshold(nfa, n_pulse)

  if n_pulse == 1:
    pd = np.exp(-vt / (1 + snr))
  else:
    pd = 1 - gammainc(n_pulse-1, vt) + \
        (1 + 1/(n_pulse*snr))**(n_pulse - 1) * \
        gammainc(n_pulse-1, vt / (1 + 1/(n_pulse*snr))) * \
        np.exp(-vt / (1 + n_pulse*snr))

  return pd

def pd_swerling2(pfa: float, n_pulse: float, snr_db: float) -> float:
  """
  Compute the probability of detection for a swerling 2 target for non-coherently integrated pulses.

  See: Mahafza 2013

  Parameters
  ----------
  pfa : float
      Probability of false alarm
  n_pulse : float
      Number of pulses
  snr_db : float
      Single-pulse SNR in dB
  """
  snr = 10**(snr_db/10)
  nfa = n_pulse * np.log(2) / pfa
  vt = threshold(nfa, n_pulse)

  if n_pulse <= 50:
    pd = 1 - gammainc(n_pulse, vt/(1 + snr))
  else:
    C3 = -1 / (3 * np.sqrt(n_pulse))
    C4 = 1 / (4 * n_pulse)
    C6 = C3**2 / 2
    w = np.sqrt(n_pulse) * (1 + snr)

    V = (vt - n_pulse*(1 + snr)) / w
    pd = 0.5*erfc(V / np.sqrt(2)) - \
        (np.exp(-V**2/2) / np.sqrt(2*np.pi)) * \
        (C3*(V**2 - 1) + C4*V*(3-V**2) - C6*V*(V**4 - 10*V**2 + 15))

  return pd

def pd_swerling3(pfa: float, n_pulse: float, snr_db: float) -> float:
  """
  Compute the probability of detection for a swerling 3 target for non-coherently integrated pulses.


  Parameters
  ----------
  pfa : float
      Probability of false alarm
  n_pulse : float
      Number of pulses
  snr_db : float
      Single-pulse SNR in dB
  """
  
  snr = 10**(snr_db/10)
  nfa = n_pulse * np.log(2) / pfa
  vt = threshold(nfa, n_pulse)
  
  pd = (1 + 2/(n_pulse*snr))**(n_pulse-2) * \
    (1 + (vt / (1 + n_pulse*snr/2)) - 2*(n_pulse-2)/(n_pulse*snr)) \
      * np.exp(-vt / (1 + n_pulse*snr/2))
  return pd
        
def pd_swerling4(pfa: float, n_pulse: float, snr_db: float) -> float:
  # TODO: This equation is computationally complex. Not sure if it's worth implementing
  pass

if __name__ == '__main__':
  t = threshold(nfa=1, n_pulse=1)
  print(pd_swerling3(pfa=1e-8, n_pulse=10, snr_db=10))