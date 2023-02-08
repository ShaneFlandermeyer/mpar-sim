import gymnasium as gym
import numpy as np

class ImageToPytorch(gym.ObservationWrapper):
  """Convert a (H,W,C) image to (C,H,W)."""
  
  def __init__(self, env: gym.Env):
    super().__init__(env)
    old_shape = self.observation_space.shape
    new_shape = (old_shape[-1], old_shape[0], old_shape[1])
    self.observation_space = gym.spaces.Box(
      low=np.reshape(self.observation_space.low, new_shape),
      high=np.reshape(self.observation_space.high, new_shape),
      shape=new_shape,
      dtype=self.observation_space.dtype
    )
    
  def observation(self, observation: np.ndarray) -> np.ndarray:
    return np.moveaxis(observation, 2, 0)