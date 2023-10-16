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
    
    self.observation_space = gym.spaces.Box(low=0, high=1, 
                                            shape=(self.pri,self.fft_size))
    self.action_space = gym.spaces.Box(low=0, high=1, shape=(1,))
  
  def reset(self, seed: int = None, options = None):
    self.interference.reset()
    obs = np.zeros(self.observation_space.shape)
    for i in range(self.pri):
      obs[i] = self.interference.step()
    return obs
  
  def step(self, action):
    obs = np.zeros(self.observation_space.shape)
    for i in range(self.pri):
      obs[i] = self.interference.step()
    
    # for isnapshot in range(self.pri):
    #   obs[snapshot]  = self.interference.step()
  
if __name__ == '__main__':
  env = gym.vector.SyncVectorEnv([lambda: SpectrumEnv()] * 4)
  
  obs = env.reset()
  plt.imshow(obs[0])
  plt.show()