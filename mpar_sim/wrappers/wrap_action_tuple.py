import gymnasium as gym

from mpar_sim.common.wrap_to_interval import wrap_to_interval


class WrapActionTuple(gym.ActionWrapper):
  """
  Wrap a tuple of scalar actions to the interval [low, high] for each action.
  """

  def __init__(self, env: gym.Env):
    super().__init__(env)

  def action(self, actions):
    new_actions = list()
    for i, action in enumerate(actions):
      new_actions.append(wrap_to_interval(action,
                                          self.action_space[i].low.item(), self.action_space[i].high.item()))
    return tuple(actions)
