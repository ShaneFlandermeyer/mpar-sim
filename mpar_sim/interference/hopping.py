import numpy as np

from mpar_sim.interference.interference import Interference


class HoppingInterference():
  def __init__(self,
               start_freq: float,
               bandwidth: float,
               duration: float,
               hop_size: float,
               channel_bw: float,
               fft_size: int,
               ):
    self.start_freq = start_freq
    self.bandwidth = bandwidth
    self.duration = duration
    self.hop_size = hop_size
    self.channel_bw = channel_bw
    self.fft_size = fft_size
    self.freq_axis = np.linspace(0, self.channel_bw, self.fft_size)
    self.reset()

  def step(self, time):
    if time - self.last_update_time >= self.duration:
      self.last_update_time = time

      # Reverse the sweep at the end of the frequency axis
      n_shift = self.direction * \
          round(self.hop_size/self.freq_axis[-1]*self.fft_size)
      self.start_ind += n_shift
      self.stop_ind += n_shift
      if self.start_ind <= 0:
        self.start_ind = 0
        self.stop_ind = int(self.bandwidth/self.freq_axis[-1]*self.fft_size)
        self.direction *= -1
      elif self.stop_ind >= self.fft_size:
        self.stop_ind = self.fft_size - 1
        self.start_ind = self.fft_size - \
            int(self.bandwidth/self.freq_axis[-1]*self.fft_size)
        self.direction *= -1

      self.state = np.logical_and(
          self.freq_axis >= self.freq_axis[self.start_ind],
          self.freq_axis <= self.freq_axis[self.stop_ind]
      )

    return self.state

  def reset(self):
    self.last_update_time = 0
    self.direction = 1
    self.start_ind = 0
    self.stop_ind = round(self.bandwidth/self.freq_axis[-1]*self.fft_size)
    self.is_active = True

    # TODO: Randomize this guy
    # self.start_freq = np.random.choice(np.arange(5))*self.bandwidth
    # self.direction = np.random.choice([-1, 1])

    self.state = np.logical_and(
        self.freq_axis >= self.start_freq,
        self.freq_axis <= self.start_freq + self.bandwidth
    )
