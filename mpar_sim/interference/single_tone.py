import numpy as np


class SingleToneInterferer():
  def __init__(self,
               start_freq: float,
               bandwidth: float,
               duration: float,
               duty_cycle: float,
               ):
    self.start_freq = start_freq
    self.bandwidth = bandwidth
    self.duration = duration
    self.duty_cycle = duty_cycle

    self.last_update_time = 0
    self.state = 1

  def update_spectrogram(self,
                         spectrogram: np.ndarray,
                         freq_axis: np.ndarray,
                         start_time: float,
                         ) -> np.ndarray:
    """
    Modify the spectrogram to include the binary mask of the interferer

    Parameters
    ----------
    spectrogram : np.ndarray
        _description_
    time_axis : np.ndarray
        _description_
    freq_axis : np.ndarray
        _description_
    start_time : float
        _description_
    stop_time : float
        _description_

    Returns
    -------
    np.ndarray
        _description_
    """
    n_freq_bins = np.digitize(
        self.bandwidth, freq_axis - np.min(freq_axis))
    i_start_freq = np.digitize(
        self.start_freq, freq_axis - np.min(freq_axis), right=True)
    i_stop_freq = i_start_freq + n_freq_bins

    # Compute the number of time bins to update
    if self.state == 1:
      update_interval = self.duration * self.duty_cycle
    else:
      update_interval = self.duration * (1 - self.duty_cycle)
      
    if start_time - self.last_update_time >= update_interval:
        self.state = 1 - self.state
        self.last_update_time = start_time
    # Move the spectrogram to the current time
    spectrogram = np.roll(spectrogram, -1, axis=0)
    spectrogram[-1:, :] = 0
    spectrogram[-1, i_start_freq:i_stop_freq] = self.state

    return spectrogram

  def reset(self):
    self.last_update_time = 0
    self.state = np.random.choice([0, 1])