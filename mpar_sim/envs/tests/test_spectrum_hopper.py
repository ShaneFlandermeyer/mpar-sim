import argparse
import os
from typing import Dict
import numpy as np
import ray
from ray import tune, air
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.models import ModelCatalog
from ray.tune.logger import pretty_print
from mpar_sim.envs.spectrum_hopper import SpectrumHopper

from mpar_sim.interference.recorded import RecordedInterference
from mpar_sim.interference.hopping import HoppingInterference
from mpar_sim.models.networks.rnn import LSTMActorCritic
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.env import BaseEnv
from ray.rllib.evaluation import Episode, RolloutWorker
from ray.rllib.policy import Policy
from ray.rllib.utils.exploration import OrnsteinUhlenbeckNoise
import random
from collections import deque


class SpectrumMetricsCallbacks(DefaultCallbacks):
  def __init__(self):
      super().__init__()
      self.saa_r_moving_avg = deque(maxlen=100)
  def on_episode_start(
      self,
      *,
      worker: RolloutWorker,
      base_env: BaseEnv,
      policies: Dict[str, Policy],
      episode: Episode,
      env_index: int,
      **kwargs
  ):
    episode.user_data["bandwidth"] = []
    episode.user_data["collision"] = []
    episode.user_data["missed"] = []
    episode.user_data["bandwidth_std"] = []
    episode.user_data["center_freq_std"] = []
    episode.user_data["bw_diff"] = []
    episode.user_data["fc_diff"] = []
    episode.user_data["saa_reward"] = []
    episode.user_data["saa_bandwidth"] = []
    episode.user_data["saa_missed"] = []
    episode.user_data["saa_collision"] = []
    episode.user_data["saa_bw_diff"] = []
    episode.user_data["saa_fc_diff"] = []

  def on_episode_step(
      self,
      *,
      worker: RolloutWorker,
      base_env: BaseEnv,
      policies: Dict[str, Policy],
      episode: Episode,
      env_index: int,
      **kwargs
  ):
    info = episode.last_info_for()
    episode.user_data["bandwidth"].append(info["bandwidth"])
    episode.user_data["collision"].append(info["collision"])
    episode.user_data["missed"].append(info["missed"])
    episode.user_data["bandwidth_std"].append(info["bandwidth_std"])
    episode.user_data["center_freq_std"].append(info["center_freq_std"])
    episode.user_data["bw_diff"].append(info["bw_diff"])
    episode.user_data["fc_diff"].append(info["fc_diff"])
    episode.user_data["saa_reward"].append(info["saa_reward"])
    episode.user_data["saa_bandwidth"].append(info["saa_bandwidth"])
    episode.user_data["saa_missed"].append(info["saa_missed"])
    episode.user_data["saa_collision"].append(info["saa_collision"])
    episode.user_data["saa_bw_diff"].append(info["saa_bw_diff"])
    episode.user_data["saa_fc_diff"].append(info["saa_fc_diff"])
    

  def on_episode_end(
      self,
      *,
      worker: RolloutWorker,
      base_env: BaseEnv,
      policies: Dict[str, Policy],
      episode: Episode,
      env_index: int,
      **kwargs
  ):
    bandwidth = np.mean(episode.user_data["bandwidth"])
    collision = np.mean(episode.user_data["collision"])
    missed = np.mean(episode.user_data["missed"])
    bw_std = np.mean(episode.user_data["bandwidth_std"])
    fc_std = np.mean(episode.user_data["center_freq_std"])
    bw_diff = np.mean(episode.user_data["bw_diff"])
    fc_diff = np.mean(episode.user_data["fc_diff"])
    saa_r = np.sum(episode.user_data["saa_reward"])
    saa_bw = np.mean(episode.user_data["saa_bandwidth"])
    saa_missed = np.mean(episode.user_data["saa_missed"])
    saa_collision = np.mean(episode.user_data["saa_collision"])
    saa_bw_diff = np.mean(episode.user_data["saa_bw_diff"])
    saa_fc_diff = np.mean(episode.user_data["saa_fc_diff"])
    self.saa_r_moving_avg.append(saa_r)
    
    episode.custom_metrics["bandwidth"] = bandwidth
    episode.custom_metrics["collision"] = collision
    episode.custom_metrics["missed"] = missed
    episode.custom_metrics["bandwidth_std"] = bw_std
    episode.custom_metrics["center_freq_std"] = fc_std
    episode.custom_metrics["bw_diff"] = bw_diff
    episode.custom_metrics["fc_diff"] = fc_diff
    episode.custom_metrics["saa_reward"] = np.mean(self.saa_r_moving_avg)
    episode.custom_metrics["saa_bandwidth"] = np.mean(saa_bw)
    episode.custom_metrics["saa_missed"] = np.mean(saa_missed)
    episode.custom_metrics["saa_collision"] = np.mean(saa_collision)
    episode.custom_metrics["saa_bw_diff"] = np.mean(saa_bw_diff)
    episode.custom_metrics["saa_fc_diff"] = np.mean(saa_fc_diff)


def get_cli_args():
  """Create CLI parser and return parsed arguments"""
  parser = argparse.ArgumentParser()

  # general args
  parser.add_argument("--exp-name", type=str)
  parser.add_argument("--exp-dir", type=str,
                      default="/home/shane/data/trs_2023")
  parser.add_argument("--num-cpus", type=int, default=31)
  parser.add_argument("--num-workers", type=int, default=9)
  parser.add_argument("--num-envs-per-worker", type=int, default=1)
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
      "--stop-iter",
      type=int,
      default=600,
      help="Number of iterations to train.",
  )

  args = parser.parse_args()
  print(f"Running with following CLI args: {args}")
  return args


if __name__ == '__main__':
  seed = 1234
  np.random.seed(seed)
  random.seed(seed)
  args = get_cli_args()
  n_envs = args.num_workers * args.num_envs_per_worker
  horizon = 128
  train_batch_size = horizon*n_envs
  lr_schedule = [[0, 1e-3], [args.stop_timesteps, 1e-4]]

  ray.init(num_cpus=args.num_cpus)
  tune.register_env(
      "SpectrumHopper", lambda env_config: SpectrumHopper(env_config))
  ModelCatalog.register_custom_model("LSTM", LSTMActorCritic)

  # TODO: For 2.64 GHz, do 1, 3, 5% collision bw, try shifting the data to improve generalization
  n_trials = 3
  config = (
      PPOConfig()
      .environment(env="SpectrumHopper", 
                   normalize_actions=True,
                   env_config={
                       "max_steps": 2000,
                       "pri": 20,
                       "cpi_len": 128,
                    #    "max_collision_bw": 2.5/100,
                       "max_collision_bw": tune.grid_search([1/100, 5/100, 10/100]), 
                       "gamma_state": 0.5,
                       "min_bandwidth": 0.1,
                   })
      .resources(
          num_gpus=1,
      )
      .training(
          train_batch_size=train_batch_size,
          model={
              "custom_model": "LSTM",
              "fcnet_hiddens": [128, 96],
              "lstm_cell_size": 64,
              "lstm_use_prev_action": True,
              "lstm_use_prev_reward": True,
          },
        #   lr=1e-3,
          lr_schedule=lr_schedule,
          gamma=0., # 0.,
          lambda_=0.95,
          clip_param=0.25,
          sgd_minibatch_size=train_batch_size,
          num_sgd_iter=15,
      )
      .rollouts(num_rollout_workers=args.num_workers,
                num_envs_per_worker=args.num_envs_per_worker,
                rollout_fragment_length="auto",
                enable_connectors=False,  # Needed for custom metrics cb
                )
      .framework(args.framework, eager_tracing=args.eager_tracing)
      .callbacks(SpectrumMetricsCallbacks)
  )
  exp_name = "spectrum_reward_2_64ghz-v7"
  tuner = tune.Tuner(
      "PPO",
      param_space=config,
      run_config=air.RunConfig(
          name=exp_name,
          stop={
            #   "training_iteration": args.stop_iter,
              "timesteps_total": args.stop_timesteps,
              },
          checkpoint_config=air.CheckpointConfig(
              checkpoint_at_end=True
          ),
          local_dir=args.exp_dir,
      ),
      tune_config=tune.TuneConfig(num_samples=n_trials),

  )
  results = tuner.fit()
  del tuner
  
  n_trials = 3
  config = (
      PPOConfig()
      .environment(env="SpectrumHopper", 
                   normalize_actions=True,
                   env_config={
                       "max_steps": 2000,
                       "pri": 20,
                       "cpi_len": 128,
                       "max_collision_bw": 5/100,
                       "gamma_state": 0.50,
                       "beta_distort": tune.grid_search([1, 2, 5]),
                       "min_bandwidth": 0.1,
                   })
      .resources(
          num_gpus=1,
      )
      .training(
          train_batch_size=train_batch_size,
          model={
              "custom_model": "LSTM",
              "fcnet_hiddens": [128, 96],
              "lstm_cell_size": 64,
              "lstm_use_prev_action": True,
              "lstm_use_prev_reward": True,
          },
        #   lr_schedule=lr_schedule,
          lr=3e-4,
          gamma=0.5, # 0.,
          lambda_=0.95,
          clip_param=0.25,
          sgd_minibatch_size=train_batch_size,
          num_sgd_iter=15,
      )
      .rollouts(num_rollout_workers=args.num_workers,
                num_envs_per_worker=args.num_envs_per_worker,
                rollout_fragment_length="auto",
                enable_connectors=False,  # Needed for custom metrics cb
                )
      .framework(args.framework, eager_tracing=args.eager_tracing)
      .callbacks(SpectrumMetricsCallbacks)
  )
  exp_name = "distortion_reward_2_64ghz-v7"
  tuner = tune.Tuner(
      "PPO",
      param_space=config,
      run_config=air.RunConfig(
          name=exp_name,
          stop={
            #   "training_iteration": args.stop_iter,
              "timesteps_total": args.stop_timesteps,
              },
          checkpoint_config=air.CheckpointConfig(
              checkpoint_at_end=True
          ),
          local_dir=args.exp_dir,
      ),
      tune_config=tune.TuneConfig(num_samples=n_trials),

  )
  results = tuner.fit()
