
import copy
from typing import Optional, Tuple
import mpar_sim.envs

import argparse
import os
import random
import time
from distutils.util import strtobool

import gymnasium as gym
import numpy as np
from lightning_rl.models.off_policy import SAC
from lightning_rl.common.utils import get_out_shape
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from lightning_rl.common.buffers import ReplayBuffer
from lightning_rl.models.off_policy.sac import squashed_gaussian_action
from lightning_rl.wrappers import TransposeObservation
from lightning_rl.modules.cnn import SACAECNN
import itertools

# Configuration
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def parse_args():
  # fmt: off
  parser = argparse.ArgumentParser()
  parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
      help="the name of this experiment")
  parser.add_argument("--seed", type=int, default=np.random.randint(0, 100000),
      help="seed of the experiment")
  parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
      help="if toggled, `torch.backends.cudnn.deterministic=False`")
  parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
      help="if toggled, cuda will be enabled by default")
  parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
      help="if toggled, this experiment will be tracked with Weights and Biases")
  parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
      help="whether to capture videos of the agent performances (check out `videos` folder)")
  parser.add_argument("--render", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
      help="if toggled, human render mode used")
  parser.add_argument("--eval-interval", type=int, default=10e3,
      help="Number of steps between environment evaluations")
  parser.add_argument("--n-eval-episodes", type=int, default=3,
      help="Number of evaluation episodes")

  # Algorithm specific arguments
  parser.add_argument("--env-id", type=str, 
                      default="mpar_sim/SpectrumHopperRecorded-v0",
                      help="the id of the environment")
  parser.add_argument("--total-timesteps", type=int, default=1000000,
      help="total timesteps of the experiments")
  parser.add_argument("--buffer-size", type=int, default=int(1e6),
      help="the replay memory buffer size")
  parser.add_argument("--gamma", type=float, default=0.99,
      help="the discount factor gamma")
  parser.add_argument("--tau", type=float, default=0.005,
      help="target smoothing coefficient (default: 0.005)")
  parser.add_argument("--batch-size", type=int, default=64,
      help="the batch size of sample from the reply memory")
  parser.add_argument("--exploration-noise", type=float, default=0.1,
      help="the scale of the exploration noise")
  parser.add_argument("--learning_starts", type=int, default=1e3,
      help="timestep to start learning")
  parser.add_argument("--policy-lr", type=float, default=3e-4,
      help="the learning rate of the policy network optimizer")
  parser.add_argument("--q-lr", type=float, default=1e-3,
      help="the learning rate of the Q network optimizer")
  parser.add_argument("--policy-frequency", type=int, default=2,
      help="the frequency of training the policy (delayed)")
  parser.add_argument("--target-network-frequency", type=int, default=1,
      help="the frequency of updates for the target networks")
  parser.add_argument("--noise-clip", type=float, default=0.5,
      help="noise clip parameter of the Target Policy Smoothing Regularization")
  parser.add_argument("--alpha", type=float, default=0.2,
      help="Entropy regularization coefficient.")
  parser.add_argument("--autotune", type=lambda x:bool(strtobool(x)), default=True, nargs="?", const=True,
      help="automatic tuning of the entropy coefficient")
  args = parser.parse_args()
  # fmt: on
  return args


def make_env(env_id, seed, idx, capture_video, run_name, render):
  def thunk():
    if render:
      render_mode = 'human'
    else:
      render_mode = 'rgb_array'
    env = gym.make(env_id,
                   filename='/home/shane/data/HOCAE_Snaps_bool_cleaned.dat',
                   channel_bandwidth=100e6,
                   fft_size=1024,
                   n_image_snapshots=256,
                   frame_stack=2,
                   obs_mode='spectrogram',
                   render_mode=render_mode)
    # TODO: This fails if more than two frames are stacked
    env = gym.wrappers.ResizeObservation(env, (84, 84))
    env = TransposeObservation(env, [-1, 0, 1])
    if capture_video:
      if idx == 0:
        env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
    env = gym.wrappers.TimeLimit(env, max_episode_steps=1000)
    env = gym.wrappers.RecordEpisodeStatistics(env)

    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    return env

  return thunk


def evaluate(env, agent, n_episodes):
  episode_rewards = []
  for i in range(n_episodes):
    obs, info = env.reset()
    done = False
    episode_reward = 0
    while not done:
      obs_tensor = torch.as_tensor(
          obs, dtype=torch.float32).to(agent.device)
      action, _ = agent.act(obs_tensor, deterministic=True)
      action = action.detach().cpu().numpy()
      obs, reward, terminated, truncated, _ = env.step(action)
      done = terminated or truncated
      episode_reward += reward

    episode_rewards.append(episode_reward)

  return np.mean(episode_rewards)


LOG_STD_MIN = -5
LOG_STD_MAX = 2


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
  torch.nn.init.orthogonal_(layer.weight, std)
  torch.nn.init.constant_(layer.bias, bias_const)
  return layer


class Actor(nn.Module):
  def __init__(self,
               obs_shape: Tuple[int],
               action_shape: Tuple[int],
               action_low: np.ndarray,
               action_high: np.ndarray
               ) -> None:
    super().__init__()
    self.convnet = SACAECNN(*obs_shape)
    out_shape = get_out_shape(self.convnet, obs_shape)
    self.start = layer_init(nn.Linear(np.prod(out_shape), 2), std=0.1)
    self.bw = layer_init(nn.Linear(np.prod(out_shape)+1, 2), std=0.1)

    # Action rescaling
    self.register_buffer(
        "action_scale", torch.tensor(
            (action_high - action_low) / 2.0, dtype=torch.float32))
    self.register_buffer(
        "action_bias", torch.tensor(
            (action_high + action_low) / 2.0, dtype=torch.float32))

  def forward(self, x):
    x = self.convnet(x / 255.0).view(x.shape[0], -1)
    s_mean, s_logstd = self.start(x).chunk(2, dim=-1)
    
    x = torch.cat([x, s_mean], dim=-1)
    bw_mean, bw_logstd = self.bw(x).chunk(2, dim=-1)
    
    mean = torch.cat([s_mean, bw_mean], dim=-1)
    logstd = torch.cat([s_logstd, bw_logstd], dim=-1)
    logstd = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (logstd + 1)
    std = logstd.exp()

    return squashed_gaussian_action(mean, std, self.action_scale, self.action_bias)


class Critic(nn.Module):
  def __init__(self, obs_shape: Tuple[int], action_shape: Tuple[int]):
    super().__init__()
    self.convnet = SACAECNN(*obs_shape)

    out_shape = np.prod(get_out_shape(self.convnet, obs_shape))
    self.Q1 = nn.Sequential(
        nn.Linear(out_shape + np.prod(action_shape), 256), nn.ReLU(),
        nn.Linear(256, 256), nn.ReLU(),
        nn.Linear(256, 1),
    )

    self.Q2 = nn.Sequential(
        nn.Linear(out_shape + np.prod(action_shape), 256), nn.ReLU(),
        nn.Linear(256, 256), nn.ReLU(),
        nn.Linear(256, 1),
    )

  def forward(self, obs: torch.Tensor, act: torch.Tensor):
    x = self.convnet(obs / 255.0).view(obs.shape[0], -1)
    x = torch.cat([x, act], dim=1)
    return self.Q1(x), self.Q2(x)


if __name__ == '__main__':
  args = parse_args()
  run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
  writer = SummaryWriter(f"runs/{run_name}")
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
      [make_env(args.env_id, args.seed, 0, args.capture_video, run_name, render=False)])
  eval_envs = gym.vector.SyncVectorEnv(
      [make_env(args.env_id, args.seed, 0, args.capture_video, run_name, render=True)])
  assert isinstance(envs.single_action_space,
                    gym.spaces.Box), "only continuous action space is supported"

  action_shape = envs.single_action_space.shape
  obs_shape = envs.single_observation_space.shape
  action_low = envs.single_action_space.low
  action_high = envs.single_action_space.high
  actor = Actor(obs_shape=obs_shape,
                action_shape=action_shape,
                action_low=action_low,
                action_high=action_high).to(device)
  critic = Critic(obs_shape=obs_shape, action_shape=action_shape).to(device)
  critic_target = copy.deepcopy(critic).to(device)
  sac = SAC(actor=actor,
            critic=critic,
            critic_target=critic_target,
            device=device,
            action_shape=action_shape,
            obs_shape=obs_shape,
            gamma=args.gamma,
            tau=args.tau,
            init_temperature=1,
            actor_lr=args.policy_lr,
            critic_lr=args.q_lr,
            alpha_lr=args.q_lr,
            )

  # Create the replay buffer
  rb = ReplayBuffer(args.buffer_size)
  rb.create_buffer('observations', sac.obs_shape,
                   envs.single_observation_space.dtype)
  rb.create_buffer('next_observations', sac.obs_shape,
                   envs.single_observation_space.dtype)
  rb.create_buffer('actions', sac.action_shape,
                   envs.single_action_space.dtype)
  rb.create_buffer('rewards', (1,), np.float32)
  rb.create_buffer('dones', (1,), bool)
  rb.create_buffer('infos', (1,), dict)

  start_time = time.time()

  # TRY NOT TO MODIFY: start the game
  obs, info = envs.reset(seed=args.seed)
  for global_step in range(args.total_timesteps):
    # ALSO LOGIC: put action logic here
    if global_step < args.learning_starts:
      actions = np.array([envs.single_action_space.sample()
                         for _ in range(envs.num_envs)])
    else:
      obs_tensor = torch.FloatTensor(obs).to(sac.device)
      actions, _ = sac.act(obs_tensor, deterministic=False)
      actions = actions.detach().cpu().numpy()

    # TRY NOT TO MODIFY: Execute the game and log data
    next_obs, rewards, terminated, truncated, infos = envs.step(actions)
    dones = terminated
    
    # TRY NOT TO MODIFY: record rewards for plotting purposes
    # Only print when at least 1 env is done
    if "final_info" in infos:
      mean_episodic_return = np.mean(
          [info["episode"]["r"] for info in infos["final_info"] if info is not None])
      mean_episodic_length = np.mean(
          [info["episode"]["l"] for info in infos["final_info"] if info is not None])
      print(
          f"global_step={global_step}, mean_reward={mean_episodic_return:.2f}, mean_length={mean_episodic_length:.2f}")
      writer.add_scalar("charts/episode_reward",
                        mean_episodic_return, global_step)
      writer.add_scalar("charts/episode_length",
                        mean_episodic_length, global_step)

    # TRY NOT TO MODIFY: save data to reply buffer; handle `terminal_observation`
    real_next_obs = next_obs.copy()
    for idx, d in enumerate(terminated | truncated):
      if d:
        real_next_obs[idx] = infos["final_observation"][idx]

    rb.add(observations=obs,
           next_observations=real_next_obs,
           actions=actions,
           rewards=rewards,
           dones=dones,
           infos=infos)

    # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
    obs = next_obs

    # ALGO LOGIC: training
    if global_step > args.learning_starts:
      train_info = {}
      # Sample the replay buffer and convert to tensors
      data = rb.sample(args.batch_size)
      observations = torch.Tensor(data.observations).to(device)
      next_observations = torch.Tensor(data.next_observations).to(device)
      actions = torch.Tensor(data.actions).to(device)
      rewards = torch.Tensor(data.rewards).to(device)
      # TODO: Necessary or helpful?
      # rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
      dones = torch.Tensor(data.dones).to(device)

      # Train critic
      critic_info = sac.update_critic(obs=observations,
                                      actions=actions,
                                      rewards=rewards,
                                      next_obs=next_observations,
                                      dones=dones)
      train_info.update(critic_info)

      # Train actor (every policy_frequency steps)
      if global_step % args.policy_frequency == 0:
        for _ in range(args.policy_frequency):
          actor_info = sac.update_actor(obs=observations)
          alpha_info = sac.update_alpha(obs=observations)
          train_info.update(actor_info)
          train_info.update(alpha_info)

      # update the target networks
      if global_step % args.target_network_frequency == 0:
        sac.soft_target_update()

      # Write statistics to tensorboard
      if global_step % 100 == 0:
        for key, value in train_info.items():
          writer.add_scalar(key, value, global_step)
        # writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("train/SPS", int(global_step /
                          (time.time() - start_time)), global_step)

      if global_step % args.eval_interval == 0:
        r = evaluate(eval_envs, sac, args.n_eval_episodes)
        print("Evaluation reward:", r)
        writer.add_scalar("train/eval_reward", r, global_step)
