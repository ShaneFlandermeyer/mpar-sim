from abc import ABCMeta
import gym
from matplotlib import pyplot as plt
import numpy as np
from mpar_sim.interference.recorded import RecordedInterference

class PRI(gym.Space):
  def __init__(self):
    super().__init__()
    # self.data = data
    
  
    

class SpectrumEnv(gym.Env):
  """
  This is a modified version of the environment I created for the original paper submission. Here, 
  """
  metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
  
  def __init__(self, render_mode=None):
    # Parameters
    self.fft_size = 1024
    self.pri = 10
    
    self.interference = RecordedInterference(
        "/home/shane/data/hocae_snaps_2_4_cleaned_10_0.dat", self.fft_size)
    self.freq_axis = np.linspace(0, 1, self.fft_size)
    
    self.observation_space = gym.spaces.Box(low=0, high=1, 
                                            shape=(self.pri, self.fft_size))
    self.action_space = gym.spaces.Box(low=0, high=1, shape=(2,))
  
  def reset(self, seed: int = None, options = None):
    # Counters
    self.pulse_count = 0
    
    self.interference.reset()
    obs = np.zeros(self.observation_space.shape)
    for i in range(self.pri):
      obs[i] = self.interference.step()
    return obs
  
  def step(self, action: np.ndarray):
    obs = np.zeros(self.observation_space.shape)
    for i in range(self.pri):
      obs[i] = self.interference.step()
      
    # TODO: Compute reward
    start_freq = action[0]
    stop_freq = max(action[1], start_freq)
    bandwidth = stop_freq - start_freq
    fc = start_freq + bandwidth / 2
    spectrum = np.logical_and(
      self.freq_axis >= start_freq, self.freq_axis <= stop_freq)
    collision_bw = np.count_nonzero(spectrum == obs[0]) / self.fft_size
    if collision_bw < 0.1:
      reward = bandwidth
    else:
      reward = 0
    
    self.pulse_count += 1
    terminated = False
    truncated = False
    
    done = terminated or truncated
    info = {}
    return obs, reward, done, info
    
  
if __name__ == '__main__':
  env = gym.vector.SyncVectorEnv([lambda: SpectrumEnv()])
  
  obs = env.reset()
  obs, reward, done, info = env.step(np.array([[0, 1]]))
  print(obs.shape)