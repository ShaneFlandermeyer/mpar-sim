import numpy as np
import matplotlib.pyplot as plt


def lfm(bandwidth: float,
        pulsewidth: float,
        samp_rate: float,
        start_freq: float,
        prf: float = None,
        ) -> np.ndarray:
  """
  Generate a linear frequency modulated (LFM) waveform.
  Parameters
  ----------
  bandwidth : float
      Sweep bandwidth (Hz)
  pulsewidth : float
      Pulse duration (s)
  samp_rate : float
      Sample rate (samples/s)
  start_freq : float
      Starting frequency of the sweep(Hz)
  prf : float, optional
      Pulse repetition frequency (PRF). If this is None, only the nonzero waveform samples are returned, by default None
  Returns
  -------
  np.ndarray
      Waveform samples
  """
  # Create the base waveform
  ts = 1.0 / samp_rate
  n_samp_pulse = int(round(pulsewidth * samp_rate))
  t = np.arange(0, n_samp_pulse) * ts
  phase = 2 * np.pi * (start_freq * t + bandwidth / (2*pulsewidth) * t**2)
  x = np.exp(1j * phase)
  
  # If no PRF is requested, return the base waveform
  if not prf:
    return x
  # Zero pad to the PRF
  n_samp_pri = int(round(samp_rate / prf))
  return np.pad(x,
                 pad_width=(0, n_samp_pri - n_samp_pulse),
                 mode='constant', constant_values=0)