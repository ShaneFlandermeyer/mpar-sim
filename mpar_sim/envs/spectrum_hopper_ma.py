from ray.tune.registry import get_trainable_cls
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray import air, tune
import ray
import os
import argparse
import functools
import ray.rllib.algorithms.ppo as ppo


import gymnasium as gym
import numpy as np
from gymnasium.spaces import Discrete

from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector, wrappers

from mpar_sim.interference.hopping import HoppingInterference


class MultiAgentSpectrum(MultiAgentEnv):
  def __init__(self, config={}):
    self.fft_size = config.get("fft_size", 1024)
    self.channel_bw = config.get("channel_bw", 1)
    self.freq_axis = np.linspace(0, self.channel_bw, self.fft_size)
    self.interference = HoppingInterference(
        bandwidth=0.2*self.channel_bw,
        duration=1,
        hop_size=0.2*self.channel_bw,
        channel_bw=self.channel_bw,
        fft_size=self.fft_size,
    )

    self.agents = ["start", "bw"]
    self._agent_ids = set(self.agents)

    # Provide full (preferred format) observation- and action-spaces as Dicts
    # mapping agent IDs to the individual agents' spaces.
    self._obs_space_in_preferred_format = True
    self.observation_space = gym.spaces.Dict(
        {
            "start": gym.spaces.Box(low=0.0, high=1.0, shape=(self.fft_size,)),
            "bw": gym.spaces.Box(low=0.0, high=1.0, shape=(self.fft_size+1,)),
        }
    )
    self._action_space_in_preferred_format = True
    self.action_space = gym.spaces.Dict(
        {
            "start": gym.spaces.Box(low=0.0, high=1.0),
            "bw": gym.spaces.Box(low=0.0, high=1.0)
        }
    )

    super().__init__()

  def reset(self, *, seed=None, options=None):
    # Reset 
    self.time = 0
    self.interference.reset()

    # Reset environment 
    self.rewards = {agent: 0 for agent in self.agents}
    self._cumulative_rewards = {agent: 0 for agent in self.agents}
    self.terminations = {agent: False for agent in self.agents}
    self.truncations = {agent: False for agent in self.agents}
    self.terminations["__all__"] = self.truncations["__all__"] = False
    self.infos = {agent: {} for agent in self.agents}
    self.state = {
        agent: self.action_space[agent].sample()
        for agent in self.agents
    }
    self.observations = {
        agent: self._get_obs(agent) for agent in self.agents
    }
    # Select the starting agent and return its info
    self.agent_ind = 0
    self.agent_selection = self.agents[self.agent_ind]
    obs = {self.agent_selection: self.observations[self.agent_selection]}
    infos = {self.agent_selection: self.infos[self.agent_selection]}
    return obs, infos 

  def step(self, action):
    # Extract the action for the current agent
    agent = self.agent_selection
    action = action[agent]

    # the agent which stepped last had its _cumulative_rewards accounted for
    # (because it was returned by last()), so the _cumulative_rewards for this
    # agent should start again at 0
    self._cumulative_rewards[agent] = 0

    # stores action of current agent
    self.state[self.agent_selection] = action

    # TODO: Update the interference before computing the reward
    if self.agent_ind == len(self.agents) - 1:
      self.time += 1
      interference_state = self.interference.step(self.time)
      # TODO: Try group reward before dividing into individual components
      reward = self._get_reward(interference_state)
      self.rewards[self.agents[0]
                   ], self.rewards[self.agents[1]] = reward, reward

      # The truncations dictionary must be updated for all players.
      MAX_NUM_ITERS = 250
      self.truncations["__all__"] = self.time >= MAX_NUM_ITERS

      # observe the current state
      for agent in self.agents:
        self.observations[agent] = self._get_obs(agent)
    else:
      self.observations[self.agents[1 - self.agent_ind]][-1] = action
      # no rewards are allocated until both players give an action
      for agent in self.agents:
        self.rewards[agent] = 0

    # selects the next agent.
    self.agent_ind = (self.agent_ind + 1) % len(self.agents)
    self.agent_selection = self.agents[self.agent_ind]
    # Adds .rewards to ._cumulative_rewards
    for agent in self.agents:
      self._cumulative_rewards[agent] += self.rewards[agent]

    # Compile step output dicts
    obs = {self.agent_selection: self.observations[self.agent_selection]}
    rewards = self.rewards
    terminations = self.terminations
    truncations = self.truncations
    infos = self.infos[agent]
    return obs, rewards, terminations, truncations, infos

  def _get_obs(self, agent):
    spectrum = np.logical_and(
        self.freq_axis >= self.interference.start_freq,
        self.freq_axis <= self.interference.start_freq + self.interference.bandwidth)
    if agent == "start":
      obs = spectrum
    else:
      obs = np.concatenate((spectrum, self.state['start']))
    return obs.astype(self.observation_space[agent].dtype)

  def _get_reward(self, interference):
    start_freq = self.state['start'].item()*self.channel_bw
    stop_freq = start_freq + self.state['bw'].item()*self.channel_bw
    radar_spectrum = np.logical_and(
        self.freq_axis >= start_freq,
        self.freq_axis <= np.clip(stop_freq, start_freq, self.channel_bw))

    n_radar_bins = np.sum(radar_spectrum)
    n_collisions = np.sum(np.logical_and(radar_spectrum, interference))
    n_tol = 0
    if n_collisions <= n_tol:
      reward = n_radar_bins / self.fft_size
    else:
      reward = n_radar_bins / self.fft_size / (n_collisions - n_tol)

    return reward


def get_cli_args():
    """Create CLI parser and return parsed arguments"""
    parser = argparse.ArgumentParser()

    # general args
    parser.add_argument(
        "--run", type=str, default="PPO", help="The RLlib-registered algorithm to use."
    )
    parser.add_argument("--num-cpus", type=int, default=0)
    parser.add_argument(
        "--framework",
        choices=["tf", "tf2", "torch"],
        default="torch",
        help="The DL framework specifier.",
    )
    parser.add_argument("--eager-tracing", action="store_true")
    parser.add_argument(
        "--stop-iters", type=int, default=100, help="Number of iterations to train."
    )
    parser.add_argument(
        "--stop-timesteps",
        type=int,
        default=1_000_000,
        help="Number of timesteps to train.",
    )
    parser.add_argument(
        "--stop-reward",
        type=float,
        default=600.0,
        help="Reward at which we stop training.",
    )
    parser.add_argument(
        "--local-mode",
        action="store_true",
        help="Init Ray in local mode for easier debugging.",
    )

    args = parser.parse_args()
    print(f"Running with following CLI args: {args}")
    return args


if __name__ == "__main__":
    args = get_cli_args()

    ray.init(num_cpus=args.num_cpus or None, local_mode=args.local_mode)

    stop = {
        # "training_iteration": args.stop_iters,
        "timesteps_total": args.stop_timesteps,
        "episode_reward_mean": args.stop_reward,
    }

    # TODO (Artur): in PPORLModule vf_share_layers = True is broken in tf2. fix it.
    vf_share_layers = not bool(os.environ.get("RLLIB_ENABLE_RL_MODULE", False))
    config = (
        get_trainable_cls(args.run)
        .get_default_config()
        .environment(env=MultiAgentSpectrum)
        .resources(
            # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
            # num_gpus=int(os.environ.get("RLLIB_NUM_GPUS", "0")),
            num_gpus=0,
        )
        .training(
          train_batch_size=1024, model={
            "vf_share_layers": vf_share_layers, 
            "use_lstm": True,
            "lstm_cell_size": 64,
            },
          gamma=0,
          )
        .rollouts(num_rollout_workers=8, rollout_fragment_length="auto")
        .framework(args.framework, eager_tracing=args.eager_tracing)
        .multi_agent(
            # Use a simple set of policy IDs. Spaces for the individual policies
            # will be inferred automatically using reverse lookup via the
            # `policy_mapping_fn` and the env provided spaces for the different
            # agents. Alternatively, you could use:
            # policies: {main0: PolicySpec(...), main1: PolicySpec}
            policies={"start", "bw"},
            # Simple mapping fn, mapping agent0 to main0 and agent1 to main1.
            policy_mapping_fn=(lambda aid, episode, worker, **kw: aid),
        )
    )

    results = tune.Tuner(
        args.run,
        run_config=air.RunConfig(
            stop=stop,
        ),
        param_space=config,
    ).fit()

    if not results:
        raise ValueError(
            "No results returned from tune.run(). Something must have gone wrong."
        )
    ray.shutdown()