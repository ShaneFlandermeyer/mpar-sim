import numpy as np

from mpar_sim.interference.interference import Interference


class SingleToneInterference(Interference):
  def __init__(self,
               bandwidth: float,
               duration: float,
               duty_cycle: float,
               channel_bw: float,
               fft_size: float,
               start_freq: float = 0,
               ):
    self.start_freq = start_freq
    self.bandwidth = bandwidth
    self.duration = duration
    self.duty_cycle = duty_cycle
    self.channel_bw = channel_bw
    self.fft_size = fft_size

    self.freq_axis = np.linspace(0, self.channel_bw, self.fft_size)
    
    self.spectrum = np.logical_and(
      self.freq_axis >= self.start_freq,
      self.freq_axis <= self.start_freq + self.bandwidth)
    self.reset()
    
  def step(self, time):
    if self.is_active:
      update_interval = self.duration * self.duty_cycle
    else:
      update_interval = self.duration * (1 - self.duty_cycle)
      
    if time - self.last_update_time >= update_interval and self.duty_cycle < 1:
      self.is_active = ~self.is_active
      self.last_update_time = time
    
    self.state = self.spectrum if self.is_active else np.zeros_like(self.spectrum)
    return self.state

  def reset(self):
    self.last_update_time = 0
    self.is_active = True
    self.state = self.spectrum if self.is_active else np.zeros_like(self.spectrum)
    
