import gymnasium as gym
import numpy as np

class SquashAction(gym.ActionWrapper):
  """
  Squash the input action to be within the range of the action space using a scaled tanh function.
  """
  def __init__(self, env):
    assert isinstance(env.action_space, gym.spaces.Box)
    super().__init__(env)
    
  def action(self, action):
      tanh = np.tanh(action)
      min_action = self.action_space.low
      max_action = self.action_space.high
      return (tanh + 1) / 2 * (max_action - min_action) + min_action