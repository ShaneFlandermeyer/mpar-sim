import gymnasium as gym
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

from mpar_sim.interference.hopping import HoppingInterference
from mpar_sim.interference.single_tone import SingleToneInterference
from mpar_sim.interference.recorded import RecordedInterference
from ray.tune.logger import pretty_print


class SpectrumHopper(gym.Env):
  def __init__(self, config):
    super().__init__()
    # TODO: This is a wrapper
    self.max_timesteps = 100
    self.fft_size = 1024
    self.channel_bw = 1
    self.freq_axis = np.linspace(0, self.channel_bw, self.fft_size)
    self.interference = HoppingInterference(
        start_freq=np.min(self.freq_axis),
        bandwidth=0.2*self.channel_bw,
        duration=1,
        hop_size=0.2*self.channel_bw,
        channel_bw=self.channel_bw,
        fft_size=self.fft_size
    )

    self.observation_space = gym.spaces.Box(
        low=0.0, high=1.0, shape=(self.fft_size,))
    self.action_space = gym.spaces.Box(low=0.0, high=1.0, shape=(2,))

  def reset(self, *, seed=None, options=None):
    self.time = 0
    self.interference.reset()
    obs = self.interference.state
    info = {}
    return obs, info

  def step(self, action):
    self.interference.step(self.time)
    reward = self._get_reward(action)
    self.time += 1
    terminated = False
    truncated = self.time >= self.max_timesteps
    obs = self.interference.state
    info = {}

    return obs, reward, terminated, truncated, info

  def _get_reward(self, action):
    start_freq = action[0]*self.channel_bw
    stop_freq = action[1]*self.channel_bw
    radar_spectrum = np.logical_and(
        self.freq_axis >= start_freq,
        self.freq_axis <= np.clip(stop_freq, start_freq, self.channel_bw))

    n_radar_bins = np.count_nonzero(radar_spectrum)
    n_collisions = np.count_nonzero(np.logical_and(
        radar_spectrum, self.interference.state))
    n_tol = 0
    reward = n_radar_bins / self.fft_size if n_collisions <= n_tol else 0

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
    parser.add_argument("--num-cpus", type=int, default=6)
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
  
if __name__ == '__main__':
  args = get_cli_args()

  ray.init(num_cpus=args.num_cpus or None)
  config = (
      get_trainable_cls(args.run)
      .get_default_config()
      .environment(env=SpectrumHopper, normalize_actions=True)
      .resources(
          # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
          num_gpus=int(os.environ.get("RLLIB_NUM_GPUS", "0")),
          # num_gpus=1,
      )
      .training(
        train_batch_size=max(args.num_cpus,1)*256, 
        model={ 
            # "fcnet_hiddens": [128, 128],
            # "lstm_cell_size": 128,
            "use_lstm": True,
            "max_seq_len": 10,
            
            # "use_attention": True,
            # "attention_num_transformer_units": 2,
            # "attention_dim": 128,
            # "attention_num_heads": 4,
            # "attention_head_dim": 32,
            # "max_seq_len": 64,
            # "attention_memory_inference": 64,
            # "attention_memory_training": 64,
            },
          gamma=0.9,
          sgd_minibatch_size=256,
          num_sgd_iter=10,
          lr=3e-4,
        )
      .rollouts(num_rollout_workers=args.num_cpus, rollout_fragment_length="auto")
      .framework(args.framework, eager_tracing=args.eager_tracing)
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
  print("Finished training. Running manual test/inference loop.")
  # env = 
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