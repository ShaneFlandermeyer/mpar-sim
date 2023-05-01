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

      # Reverse sweep direction at the ends of the channel
      end_freq = self.start_freq + self.bandwidth
      step = self.hop_size*self.direction
      if end_freq > np.max(self.freq_axis) or \
              self.start_freq + step < np.min(self.freq_axis):
        self.direction *= -1

      self.start_freq += self.hop_size*self.direction
      
      self.state = np.logical_and(
        self.freq_axis >= self.start_freq,
        self.freq_axis <= self.start_freq + self.bandwidth
      )
      
    return self.state
      
  def reset(self):
    self.last_update_time = 0
    self.direction = 1
    self.is_active = True
    
    self.state = np.logical_and(
        self.freq_axis >= self.start_freq,
        self.freq_axis <= self.start_freq + self.bandwidth
      )