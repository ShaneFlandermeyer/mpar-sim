import gymnasium as gym
import numpy as np

class SqueezeImage(gym.ObservationWrapper):
  """Remove singleton dimensions from observed images."""
  
  def __init__(self, env: gym.Env):
    super().__init__(env)
    new_shape = self.observation_space.low.squeeze().shape
    self.observation_space = gym.spaces.Box(
      low=self.observation_space.low.squeeze(),
      high=self.observation_space.low.squeeze(),
      shape=new_shape,
      dtype=self.observation_space.dtype
    )
    
  def observation(self, observation: np.ndarray) -> np.ndarray:
    return observation.squeeze()