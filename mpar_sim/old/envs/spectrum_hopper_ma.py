import itertools
from ray.tune.registry import get_trainable_cls
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray import air, tune
import ray
import os
import argparse
import functools
import ray.rllib.algorithms.ppo as ppo
from ray.rllib.utils.exploration import OrnsteinUhlenbeckNoise, GaussianNoise


import gymnasium as gym
import numpy as np
from gymnasium.spaces import Discrete

from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector, wrappers

from mpar_sim.interference.hopping import HoppingInterference
from mpar_sim.interference.single_tone import SingleToneInterference
from mpar_sim.interference.recorded import RecordedInterference
from ray.tune.logger import pretty_print

class MultiAgentSpectrum(MultiAgentEnv):
  def __init__(self, config={}):
    super().__init__()
    self.max_timesteps = config.get("max_timesteps", 200)
    self.fft_size = config.get("fft_size", 1024)
    self.channel_bw = config.get("channel_bw", 1)
    self.freq_axis = np.linspace(0, self.channel_bw, self.fft_size)
    self.interference = HoppingInterference(
        start_freq=np.min(self.freq_axis),
        bandwidth=0.2*self.channel_bw,
        duration=1,
        # duty_cycle=1,
        hop_size=0.2*self.channel_bw,
        channel_bw=self.channel_bw,
        fft_size=self.fft_size
    )
    # self.interference = RecordedInterference(
    #   filename="/home/shane/data/HOCAE_Snaps_bool.dat",
    #   fft_size=1024
    #   )
    

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

    if self.agent_ind == len(self.agents) - 1:
      self.interference.step(self.time)
      reward = self._get_reward()
      self.rewards[self.agents[0]
                   ], self.rewards[self.agents[1]] = reward, reward

      # The truncations dictionary must be updated for all players.
      self.time += 1
      # TODO: Was working before this change. 
      self.terminations["__all__"] = self.truncations["__all__"] = self.time >= self.max_timesteps

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
    interference = self.interference.state
    if agent == "start":
      obs = interference
    elif agent == "bw":
      obs = np.concatenate((interference, self.state['start']))
    return obs.astype(self.observation_space[agent].dtype)

  def _get_reward(self):
    interference = self.interference.state
    # widest = self._get_widest(interference)
    # n_widest = widest[1] - widest[0]
    
    start_freq = self.state['start'].item()*self.channel_bw
    stop_freq = start_freq + self.state['bw'].item()*self.channel_bw
    radar_spectrum = np.logical_and(
        self.freq_axis >= start_freq,
        self.freq_axis <= np.clip(stop_freq, start_freq, self.channel_bw))

    n_radar_bins = np.sum(radar_spectrum)
    n_collisions = np.sum(np.logical_and(radar_spectrum, interference))
    n_tol = 0
    if n_collisions <= n_tol:
      # reward = (n_radar_bins - n_widest) / self.fft_size
      reward = n_radar_bins / self.fft_size
    else:
      reward = 0

    return reward
  
  def _get_widest(self, spectrum: np.ndarray):
    # Group consecutive bins by value
    gap_widths = np.array([[x[0], len(list(x[1]))]
                          for x in itertools.groupby(spectrum)])
    vals = gap_widths[:, 0]
    widths = gap_widths[:, 1]
    starts = np.cumsum(widths) - widths

    # Compute the start frequency and width of the widest gap (in bins)
    open = (vals == 0)
    if not np.any(open):
      return np.array([0, 0])
    istart_widest = np.argmax(widths[open])
    widest_start = starts[open][istart_widest]
    widest_bw = widths[open][istart_widest]
    widest_stop = widest_start + widest_bw

    widest_start = widest_start
    widest_stop = widest_stop
    return np.array([widest_start, widest_stop])


def get_cli_args():
    """Create CLI parser and return parsed arguments"""
    parser = argparse.ArgumentParser()

    # general args
    parser.add_argument(
        "--run", type=str, default="PPO", help="The RLlib-registered algorithm to use."
    )
    parser.add_argument("--num-cpus", type=int, default=14)
    parser.add_argument(
        "--framework",
        choices=["tf", "tf2", "torch"],
        default="torch",
        help="The DL framework specifier.",
    )
    parser.add_argument("--eager-tracing", action="store_true")
    parser.add_argument(
        "--stop-timesteps",
        type=int,
        default=1_000_000,
        help="Number of timesteps to train.",
    )
    parser.add_argument(
        "--stop-reward",
        type=float,
        default=500.0,
        help="Reward at which we stop training.",
    )

    args = parser.parse_args()
    print(f"Running with following CLI args: {args}")
    return args


if __name__ == "__main__":
    args = get_cli_args()

    ray.init(num_cpus=args.num_cpus or None)

    config = (
        get_trainable_cls(args.run)
        .get_default_config()
        .environment(env=MultiAgentSpectrum, normalize_actions=True)
        .exploration(
          explore=True,
        )
        .resources(
            # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
            num_gpus=int(os.environ.get("RLLIB_NUM_GPUS", "0")),
            # num_gpus=1,
        )
        .training(
          train_batch_size=max(args.num_cpus, 1)*256, 
          model={ 
            # "use_lstm": True,
            # "lstm_cell_size": 256,
            
            "use_attention": True,
            "max_seq_len": 10,
            # The number of transformer units within GTrXL.
            # A transformer unit in GTrXL consists of a) MultiHeadAttention module and
            # b) a position-wise MLP.
            "attention_num_transformer_units": 1,
            # The input and output size of each transformer unit.
            "attention_dim": 64,
            # The number of attention heads within the MultiHeadAttention units.
            "attention_num_heads": 1,
            # The dim of a single head (within the MultiHeadAttention units).
            "attention_head_dim": 32,
            # The memory sizes for inference and training.
            "attention_memory_inference": 10,
            "attention_memory_training": 10,
            # The output dim of the position-wise MLP.
            "attention_position_wise_mlp_dim": 32,
            # The initial bias values for the 2 GRU gates within a transformer unit.
            "attention_init_gru_gate_bias": 2.0,
            # Whether to feed a_{t-n:t-1} to GTrXL (one-hot encoded if discrete).
            "attention_use_n_prev_actions": 0,
            # Whether to feed r_{t-n:t-1} to GTrXL.
            "attention_use_n_prev_rewards": 0,
            },
          gamma=0.9,
          sgd_minibatch_size=256,
          num_sgd_iter=10,
          )
        .rollouts(num_rollout_workers=args.num_cpus, rollout_fragment_length="auto")
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
    
    # Training loop
    algo = config.build()
    while True:
        result = algo.train()
        print(pretty_print(result))
        # stop training if the target train steps or reward are reached
        if (
            result["timesteps_total"] >= args.stop_timesteps
            or result["episode_reward_mean"] >= args.stop_reward
        ):
            break
        
    # TODO: Evaluation
    # print("Finished training. Running manual test/inference loop.")
    # env = MultiAgentSpectrum()
    # obs, info = env.reset()
    # done = False
    # total_reward = 0
    # n_tf = config["model"]["attention_num_transformer_units"]
    # start_state = algo.get_policy("start").get_initial_state()
    # bw_state = algo.get_policy("bw").get_initial_state()
    # # run one iteration until done
    # print(f"RepeatAfterMeEnv with {config['env_config']}")
    # while not done:
    #   start_action, start_state_out, _ = algo.compute_single_action(obs["start"], start_state, policy_id="start") # TODO
    #   next_obs_start, _, _, _, _ = env.step(start_action)
    #   obs_start = next_obs_start
      
      
    ray.shutdown()
