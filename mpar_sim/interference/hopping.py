import numpy as np

from mpar_sim.interference.interference import Interference


class HoppingInterference():
  def __init__(self,
               start_freq: float,
               bandwidth: float,
               duration: float,
               hop_size: float,
               min_freq: float,
               max_freq: float
               ):
    self.start_freq = start_freq
    self.bandwidth = bandwidth
    self.duration = duration
    self.hop_size = hop_size
    self.min_freq = min_freq
    self.max_freq = max_freq

    self.last_update_time = 0
    self.direction = 1
    self.is_active = True

  def step(self, time):
    if time - self.last_update_time >= self.duration:
      self.last_update_time = time

      # Reverse sweep direction at the ends of the channel
      end_freq = self.start_freq + self.bandwidth
      step = self.hop_size*self.direction
      if end_freq > self.max_freq or \
              self.start_freq + step < self.min_freq:
        self.direction *= -1

      self.start_freq += self.hop_size*self.direction
      
  def reset(self):
    self.last_update_time = 0
    self.direction = 1
    self.is_active = True