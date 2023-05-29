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


class SpectrumMetricsCallbacks(DefaultCallbacks):
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

    episode.custom_metrics["bandwidth"] = bandwidth
    episode.custom_metrics["collision"] = collision
    episode.custom_metrics["missed"] = missed
    episode.custom_metrics["bandwidth_std"] = bw_std
    episode.custom_metrics["center_freq_std"] = fc_std


def get_cli_args():
  """Create CLI parser and return parsed arguments"""
  parser = argparse.ArgumentParser()

  # general args
  parser.add_argument("--exp-name", type=str)
  parser.add_argument("--exp-dir", type=str,
                      default="/home/shane/data/trs_2023")
  parser.add_argument("--num-cpus", type=int, default=21)
  parser.add_argument("--num-workers", type=int, default=20)
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
      default=2_000_000,
      help="Number of timesteps to train.",
  )

  args = parser.parse_args()
  print(f"Running with following CLI args: {args}")
  return args


if __name__ == '__main__':
#   random.seed(1234)
#   np.random.seed(1234)
  args = get_cli_args()
  n_envs = args.num_workers * args.num_envs_per_worker
  horizon = 128
  train_batch_size = horizon*n_envs
  lr_schedule = [[0, 8e-4], [args.stop_timesteps, 2e-4]]

  ray.init(num_cpus=args.num_cpus)
  tune.register_env(
      "SpectrumHopper", lambda env_config: SpectrumHopper(env_config))
  ModelCatalog.register_custom_model("LSTM", LSTMActorCritic)
  config = (
      PPOConfig()
      .environment(env="SpectrumHopper", normalize_actions=True,
                   env_config={
                       "max_steps": 2000,
                       "pri": 20,
                       "cpi_len": 32,
                       "min_collision_bw": 0/100,
                    #    "max_collision_bw": 5/100,
                       "max_collision_bw": tune.grid_search([1/100, 5/100, 10/100]),
                       "gamma_state": 0.5,
                       #    "beta_distort": tune.grid_search([0.0, 0.5, 1.0]),
                       #  "beta_fc": tune.grid_search([0.0, 0.5, 1.0]),
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
          lr_schedule=lr_schedule,
          gamma=0.,
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
      .debugging(seed=tune.randint(0, 10000))
      .callbacks(SpectrumMetricsCallbacks)
      .exploration(explore=OrnsteinUhlenbeckNoise)
  )

  # Tune API
  n_trials = 5
  tuner = tune.Tuner(
      "PPO",
      param_space=config,
      run_config=air.RunConfig(
          name=args.exp_name,
          stop={"timesteps_total": args.stop_timesteps},
          checkpoint_config=air.CheckpointConfig(
              checkpoint_at_end=True
          ),
          local_dir=args.exp_dir,
      ),
      tune_config=tune.TuneConfig(num_samples=n_trials),
  )
  results = tuner.fit()
#   print("Finished training. Running manual test/inference loop.")
#   best_result = results.get_best_result("episode_reward_mean", "max")
#   algo = Algorithm.from_checkpoint(best_result.checkpoint)


#   # Prepare env
#   env_config = config["env_config"]
#   env_config["render_mode"] = "human"
#   env = SpectrumHopper(env_config)
#   obs, info = env.reset()
#   done = False
#   total_reward = 0

#   # Initialize memory
#   lstm_cell_size = config["model"]["lstm_cell_size"]
#   init_state = state = [
#       np.zeros([lstm_cell_size], np.float32) for _ in range(4)]
#   prev_action = np.zeros(env.action_space.shape, np.float32)
#   prev_reward = 0
#   while not done:
#     # action = env.action_space.sample()
#     action, state, _ = algo.compute_single_action(
#         obs, state, prev_action=prev_action, prev_reward=prev_reward, explore=False)
#     obs, reward, terminated, truncated, info = env.step(action)
#     done = terminated or truncated
#     prev_action = action
#     prev_reward = reward

#     total_reward += reward
#     env.render()
#   print("Total eval. reward:", total_reward)

  ray.shutdown()
