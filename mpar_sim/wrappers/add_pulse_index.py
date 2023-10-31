import gymnasium as gym
import numpy as np

class AddPulseIndex(gym.ObservationWrapper):
  """
  Add the scalar value from the 'pulse_index' key in the info dict of the original env
  """
  def __init__(self, env):
    super().__init__(env)
    self.observation_space = gym.spaces.Box(
        low=np.concatenate((self.observation_space.low, [0])),
        high=np.concatenate((self.observation_space.high, [1])),
        dtype=np.float32
    )
  
  def step(self, action):
    observation, reward, terminated, truncated, info = self.env.step(action)
    observation = np.concatenate((observation, [info['pulse_index']]))
    return observation, reward, terminated, truncated, info
  
  def reset(self, **kwargs):
    observation, info = self.env.reset(**kwargs)
    observation = np.concatenate((observation, [0]))
    return observation, info
