import gymnasium as gym
import numpy as np

class TakeLastN(gym.ObservationWrapper):
  """Remove singleton dimensions from observed images."""
  
  def __init__(self, env: gym.Env, n: int):
    super().__init__(env)
    self.n = n
    old_shape = self.observation_space.shape
    new_shape = (n, *old_shape[1:])
    self.observation_space = gym.spaces.Box(
      low=0,
      high=1,
      shape=new_shape,
      dtype=self.observation_space.dtype
    )
  def observation(self, observation: np.ndarray) -> np.ndarray:
    return observation[-self.n:]