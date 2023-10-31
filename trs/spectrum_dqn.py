# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/dqn/#dqnpy
import argparse
import os
import random
import time
from distutils.util import strtobool

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter
import mpar_sim.envs
from mpar_sim.wrappers.first_n import TakeFirstN
from mpar_sim.wrappers.last_n import TakeLastN

ENV_KWARGS = dict(
    # dataset="/home/shane/data/hocae_snaps_2_4_cleaned_10_0.dat",
    dataset="/home/shane/data/hocae_snaps_2_64_cleaned_10_0.dat",
    pri=10,
    order="C",
    collision_weight=40,
    n_action_bins=10,
)


def parse_args():
  # fmt: off
  parser = argparse.ArgumentParser()
  parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
      help="the name of this experiment")
  parser.add_argument("--logdir", type=str, default="trs/logs/ddqn/2_64_experiment1",)
  parser.add_argument("--seed", type=int, default=np.random.randint(0, 2**32 - 1),
      help="seed of the experiment")
  parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
      help="if toggled, `torch.backends.cudnn.deterministic=False`")
  parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
      help="if toggled, cuda will be enabled by default")
  parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
      help="if toggled, this experiment will be tracked with Weights and Biases")
  parser.add_argument("--wandb-project-name", type=str, default="cleanRL",
      help="the wandb's project name")
  parser.add_argument("--wandb-entity", type=str, default=None,
      help="the entity (team) of wandb's project")
  parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
      help="whether to capture videos of the agent performances (check out `videos` folder)")
  parser.add_argument("--save-model", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
      help="whether to save model into the `runs/{run_name}` folder")
  parser.add_argument("--upload-model", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
      help="whether to upload the saved model to huggingface")
  parser.add_argument("--hf-entity", type=str, default="",
      help="the user or org name of the model repository from the Hugging Face Hub")

  # Algorithm specific arguments
  parser.add_argument("--env-id", type=str, default="mpar_sim/SpectrumEnv",
      help="the id of the environment")
  parser.add_argument("--total-timesteps", type=int, default=2_000_000,
      help="total timesteps of the experiments")
  parser.add_argument("--learning-rate", type=float, default=1e-3,
      help="the learning rate of the optimizer")
  parser.add_argument("--num-envs", type=int, default=1,
      help="the number of parallel game environments")
  parser.add_argument("--buffer-size", type=int, default=2000,
      help="the replay memory buffer size")
  parser.add_argument("--gamma", type=float, default=0.7,
      help="the discount factor gamma")
  parser.add_argument("--tau", type=float, default=1,
      help="the target network update rate")
  parser.add_argument("--target-network-frequency", type=int, default=250,
      help="the timesteps it takes to update the target network")
  parser.add_argument("--batch-size", type=int, default=128,
      help="the batch size of sample from the reply memory")
  parser.add_argument("--start-e", type=float, default=1,
      help="the starting epsilon for exploration")
  parser.add_argument("--end-e", type=float, default=0.05,
      help="the ending epsilon for exploration")
  parser.add_argument("--exploration-fraction", type=float, default=0.5,
      help="the fraction of `total-timesteps` it takes from start-e to go end-e")
  parser.add_argument("--learning-starts", type=int, default=2000,
      help="timestep to start learning")
  parser.add_argument("--train-frequency", type=int, default=10,
      help="the frequency of training")
  args = parser.parse_args()
  # fmt: on
  assert args.num_envs == 1, "vectorized envs are not supported at the moment"

  return args


def make_env(env_id, seed, idx, capture_video, run_name):
  def thunk():
    if capture_video and idx == 0:
      env = gym.make(env_id, render_mode="rgb_array",
                     seed=seed+idx, **ENV_KWARGS)
      env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
    else:
      env = gym.make(env_id, seed=seed+idx, **ENV_KWARGS)
    env = TakeLastN(env, 1)
    # env = TakeFirstN(env, 8)
    env = gym.wrappers.FlattenObservation(env)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env.action_space.seed(seed)

    return env

  return thunk


# ALGO LOGIC: initialize agent here:
class QNetwork(nn.Module):
  def __init__(self, env):
    super().__init__()
    self.network = nn.Sequential(
        nn.Linear(np.array(env.single_observation_space.shape).prod(), 128),
        nn.ReLU(),
        nn.Linear(128, 84),
        nn.ReLU(),
        nn.Linear(84, env.single_action_space.n),
    )

  def forward(self, x):
    return self.network(x)


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
  slope = (end_e - start_e) / duration
  return max(slope * t + start_e, end_e)


if __name__ == "__main__":
  import stable_baselines3 as sb3

  if sb3.__version__ < "2.0":
    raise ValueError(
        """Ongoing migration: run the following command to install the new dependencies:

poetry run pip install "stable_baselines3==2.0.0a1"
"""
    )
  args = parse_args()
  run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
  if args.track:
    import wandb

    wandb.init(
        project=args.wandb_project_name,
        entity=args.wandb_entity,
        sync_tensorboard=True,
        config=vars(args),
        name=run_name,
        monitor_gym=True,
        save_code=True,
    )
  writer = SummaryWriter(f"{args.logdir}/{run_name}")
  writer.add_text(
      "hyperparameters",
      "|param|value|\n|-|-|\n%s" % (
          "\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
  )

  # TRY NOT TO MODIFY: seeding
  random.seed(args.seed)
  np.random.seed(args.seed)
  torch.manual_seed(args.seed)
  torch.backends.cudnn.deterministic = args.torch_deterministic

  device = torch.device("cuda" if torch.cuda.is_available()
                        and args.cuda else "cpu")

  # env setup
  envs = gym.vector.SyncVectorEnv(
      [make_env(args.env_id, args.seed + i, i, args.capture_video, run_name)
       for i in range(args.num_envs)]
  )
  assert isinstance(envs.single_action_space,
                    gym.spaces.Discrete), "only discrete action space is supported"

  q_network = QNetwork(envs).to(device)
  optimizer = optim.Adam(q_network.parameters(), lr=args.learning_rate)
  target_network = QNetwork(envs).to(device)
  target_network.load_state_dict(q_network.state_dict())

  rb = ReplayBuffer(
      args.buffer_size,
      envs.single_observation_space,
      envs.single_action_space,
      device,
      handle_timeout_termination=False,
  )
  start_time = time.time()

  # TRY NOT TO MODIFY: start the game
  ep_count = 0
  obs, _ = envs.reset(seed=args.seed)
  for global_step in range(args.total_timesteps):
    # ALGO LOGIC: put action logic here
    epsilon = linear_schedule(
        args.start_e, args.end_e, args.exploration_fraction * args.total_timesteps, global_step)
    if random.random() < epsilon:
      actions = np.array([envs.single_action_space.sample()
                         for _ in range(envs.num_envs)])
    else:
      q_values = q_network(torch.Tensor(obs).to(device))
      actions = torch.argmax(q_values, dim=1).cpu().numpy()

    # TRY NOT TO MODIFY: execute the game and log data.
    next_obs, rewards, terminations, truncations, infos = envs.step(actions)

    # TRY NOT TO MODIFY: record rewards for plotting purposes
    # if "final_info" not in infos:
    #     continue
    log_interval = 50
    if np.any(np.logical_or(terminations, truncations)):
      ep_count += 1
      if ep_count % log_interval == 0:
        writer.add_scalar("charts/episodic_return",
                          infos['final_info'][0]["episode"]["r"], global_step)
        writer.add_scalar("charts/episodic_length",
                          infos['final_info'][0]["episode"]["l"], global_step)
        avg_mean_bw = np.mean(infos['final_info'][0]['mean_bw'])
        avg_mean_col = np.mean(infos['final_info'][0]['mean_collision_bw'])
        avg_mean_widest = np.mean(infos['final_info'][0]['mean_widest_bw'])
        avg_mean_missed_bw = np.mean(infos['final_info'][0]['mean_missed_bw'])
        avg_mean_bw_diff = np.mean(infos['final_info'][0]['mean_bw_diff'])
        avg_mean_fc_diff = np.mean(infos['final_info'][0]['mean_fc_diff'])
        writer.add_scalar("charts/mean_bandwidth", avg_mean_bw, global_step)
        writer.add_scalar("charts/mean_collision_bw",
                          avg_mean_col, global_step)
        writer.add_scalar("charts/mean_widest_bw",
                          avg_mean_widest, global_step)
        writer.add_scalar("charts/mean_missed_bw",
                          avg_mean_missed_bw, global_step)
        writer.add_scalar("charts/mean_bw_diff", avg_mean_bw_diff, global_step)
        writer.add_scalar("charts/mean_fc_diff", avg_mean_fc_diff, global_step)
        print(
            f"global_step={global_step}, bw={avg_mean_bw:.3f}, col={avg_mean_col:.3f}, widest={avg_mean_widest:.3f}")

    # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
    real_next_obs = next_obs.copy()
    for idx, trunc in enumerate(truncations):
      if trunc:
        real_next_obs[idx] = infos["final_observation"][idx]
    rb.add(obs, real_next_obs, actions, rewards, terminations, infos)

    # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
    obs = next_obs

    # ALGO LOGIC: training.
    if global_step > args.learning_starts:
      if global_step % args.train_frequency == 0:
        data = rb.sample(args.batch_size)
        with torch.no_grad():
          target_max, _ = target_network(data.next_observations).max(dim=1)
          td_target = data.rewards.flatten() + args.gamma * target_max * \
              (1 - data.dones.flatten())
        old_val = q_network(data.observations).gather(
            1, data.actions).squeeze()
        loss = F.mse_loss(td_target, old_val)

        # if global_step % 100 == 0:
        #   writer.add_scalar("losses/td_loss", loss, global_step)
        #   writer.add_scalar("losses/q_values",
        #                     old_val.mean().item(), global_step)
        #   print("SPS:", int(global_step / (time.time() - start_time)))
        #   writer.add_scalar("charts/SPS", int(global_step /
        #                     (time.time() - start_time)), global_step)

        # optimize the model
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

      # update target network
      if global_step % args.target_network_frequency == 0:
        for target_network_param, q_network_param in zip(target_network.parameters(), q_network.parameters()):
          target_network_param.data.copy_(
              args.tau * q_network_param.data +
              (1.0 - args.tau) * target_network_param.data
          )

  envs.close()
  writer.close()
