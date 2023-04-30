import numpy as np

from mpar_sim.interference.interference import Interference


class HoppingInterference(Interference):
  # TODO: This class appears to be broken somehow. Seems like the interference state might be incorrect?
  def __init__(self,
               bandwidth: float,
               duration: float,
               hop_size: float,
               channel_bw: float,
               fft_size: float,
               start_freq: float = 0,
               ):
    self.start_freq = start_freq
    self.bandwidth = bandwidth
    self.duration = duration
    self.hop_size = hop_size
    self.channel_bw = channel_bw
    self.fft_size = fft_size

    self.freq_axis = np.linspace(0, self.channel_bw, self.fft_size)
    self.reset(randomize=False)

  def step(self, time):
    if time - self.last_update_time >= self.duration:
      self.last_update_time = time

      # Reverse sweep direction at the ends of the channel
      end_freq = self.current_start + self.bandwidth
      step = self.hop_size*self.direction
      if end_freq > self.channel_bw or self.current_start + step < 0:
        self.direction *= -1

      # Update the current frequency range
      self.current_start += self.hop_size*self.direction
      self.current_stop = self.current_start + self.bandwidth
      self.state = np.logical_and(
          self.freq_axis >= self.current_start,
          self.freq_axis <= self.current_stop
      )
      
    return self.state

  def reset(self, randomize: bool = True):
    self.last_update_time = 0
    # TODO: Add an option to randomize these at the start
    if randomize:
      self.direction = np.random.choice([-1, 1])
      self.start_freq = np.random.uniform(0, self.channel_bw - self.bandwidth)
    else:
      self.direction = 1

    # Reset the frequency state
    self.current_start = self.start_freq
    self.current_stop = self.start_freq + self.bandwidth
    self.state = np.logical_and(
        self.freq_axis >= self.current_start,
        self.freq_axis <= self.current_stop
    )
    
    return self.state
