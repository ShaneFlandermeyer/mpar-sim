import pprint
from ray.tune.registry import get_trainable_cls
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray import air, tune
import ray
import os
import argparse
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig

import gymnasium as gym
import numpy as np

from mpar_sim.interference.hopping import HoppingInterference
from mpar_sim.interference.single_tone import SingleToneInterference


class MultiAgentSpectrum(MultiAgentEnv):
  def __init__(self, config={}):
    super().__init__()
    self.max_timesteps = config.get("max_timesteps", 250)
    self.fft_size = config.get("fft_size", 1024)
    self.channel_bw = config.get("channel_bw", 100e6)
    self.interference = config.get("interference", HoppingInterference(
        bandwidth=0.2*self.channel_bw,
        duration=1,
        hop_size=0.2*self.channel_bw,
        channel_bw=self.channel_bw,
        fft_size=self.fft_size,
    ))
    # self.interference = config.get("interference", SingleToneInterference(
    #     bandwidth=0.2*self.channel_bw,
    #     duration=2,
    #     duty_cycle=0.5,
    #     channel_bw=self.channel_bw,
    #     fft_size=self.fft_size,
    # ))
    self.freq_axis = np.linspace(0, self.channel_bw, self.fft_size)

    # Create the agents. The
    self.agents = ["start", "bw"]
    self._agent_ids = set(self.agents)
    self._obs_space_in_preferred_format = True
    self._action_space_in_preferred_format = True
    self.observation_space = gym.spaces.Dict(
        {
            "start": gym.spaces.Box(low=0.0, high=1.0, shape=(self.fft_size,)),
            "bw": gym.spaces.Box(low=0.0, high=1.0, shape=(self.fft_size+1,)),
        }
    )
    self.action_space = gym.spaces.Dict(
        {
            "start": gym.spaces.Box(low=0.0, high=1.0),
            "bw": gym.spaces.Box(low=0.0, high=1.0)
        }
    )

  def reset(self, *, seed=None, options=None):
    # Reset
    self.time = 0
    self.interference.reset(False)

    # Reset environment
    self.rewards = {agent: 0 for agent in self.agents}
    self._cumulative_rewards = {agent: 0 for agent in self.agents}
    self.terminations = {agent: False for agent in self.agents + ["__all__"]}
    self.truncations = {agent: False for agent in self.agents + ["__all__"]}
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
    active_agent = self.agent_selection
    action = action[active_agent]

    # the agent which stepped last had its _cumulative_rewards accounted for
    # (because it was returned by last()), so the _cumulative_rewards for this
    # agent should start again at 0
    self._cumulative_rewards[active_agent] = 0

    # stores action of current agent
    self.state[self.agent_selection] = action

    if self.agent_ind == len(self.agents) - 1:
      self.time += 1
      interference_state = self.interference.step(self.time)
      reward = self._get_reward(interference_state)
      for agent in self.agents:
        self.rewards[agent] += reward

      # The truncations dictionary must be updated for all players.
      self.truncations["__all__"] = self.time >= self.max_timesteps

      # observe the current state
      for agent in self.agents:
        self.observations[agent] = self._get_obs(agent)
    else:
      self.observations[active_agent][-1] = action
      # Reward is only given after the full waveform has been generated.
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
    infos = self.infos[active_agent]
    return obs, rewards, terminations, truncations, infos

  def render(self, mode="human"):
    print("Test")
    return np.zeros((64, 64, 3))

  def _get_obs(self, agent):
    spectrum = self.interference.state
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
  parser.add_argument(
      "--framework",
      choices=["tf", "tf2", "torch"],
      default="torch",
      help="The DL framework specifier.",
  )
  parser.add_argument("--eager-tracing", action="store_true")
  parser.add_argument(
      "--stop-iters", type=int, default=500, help="Number of iterations to train."
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
      default=490.0,
      help="Reward at which we stop training.",
  )

  args = parser.parse_args()
  print(f"Running with following CLI args: {args}")
  return args


if __name__ == "__main__":
  args = get_cli_args()

  ray.init()

  stop = {
      "training_iteration": args.stop_iters,
      "timesteps_total": args.stop_timesteps,
      "episode_reward_mean": args.stop_reward,
  }

  vf_share_layers = not bool(os.environ.get("RLLIB_ENABLE_RL_MODULE", False))

  config = (
      PPOConfig()
      .environment(env=MultiAgentSpectrum)
      .resources(
          num_gpus=int(os.environ.get("RLLIB_NUM_GPUS", "0")),
      )
      .training(
          train_batch_size=1024, model={
              "vf_share_layers": False,
              "use_lstm": True,
              "lstm_cell_size": 64,
          },
          gamma=0.,
      )
      .rollouts(num_rollout_workers=6, rollout_fragment_length='auto')
      .framework(args.framework, eager_tracing=args.eager_tracing)
      .multi_agent(
          policies={"start", "bw"},
          policy_mapping_fn=(lambda agent_id, episode, worker, **kw: agent_id),
      )
  )
  # algo = config.build()
  # for i in range(250):
  #   results = algo.train()
  #   del results['config']
  #   mean_reward = np.mean(results['hist_stats']['episode_reward'])
  #   print(f"{i}: {mean_reward}")
    
  results = tune.run(args.run, config=config, stop=stop)
  # results = tune.Tuner(
  #   args.run,
  #   run_config=air.RunConfig(stop=stop),
  #   param_space=config,
  # ).fit()
  # pprint.pprint(results)

  # Evaluate the agent outside the RLLib training loop
  env = MultiAgentSpectrum()
  obs, info = env.reset()
  init_state = start_state = bw_state = [
      np.zeros([config["model"]["lstm_cell_size"]],
               dtype=np.float32) for _ in range(2)
  ]
    
  total_reward = 0
  for i in range(250):
    start, start_state, _ = algo.compute_single_action(obs['start'], start_state, policy_id='start')
    obs, _, _, _, _ = env.step({'start': start})
    bw, bw_state, _ = algo.compute_single_action(obs['bw'], bw_state, policy_id='bw')
    obs, reward, terminated, truncated, info = env.step({'bw': bw})
    total_reward += reward['start']
  print(total_reward)

  ray.shutdown()