
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
from lightning_rl.models.networks.nature import NatureEncoder
from lightning_rl.models.networks.sacae import SACAEEncoder
from lightning_rl.models.networks.impala import ImpalaEncoder
from lightning_rl.common.utils import get_out_shape
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from lightning_rl.common.buffers import ReplayBuffer
from lightning_rl.models.off_policy.sac import squashed_gaussian_action

# Configuration
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class Actor(nn.Module):
  def __init__(self,
               obs_space: gym.Space,
               action_space: gym.Space,
               hidden_dim: int,
               encoder_dim: int,
               lr: float,
               action_scale: torch.Tensor,
               action_bias: torch.Tensor,
               feature_dim: int = 50,
               logstd_min: float = -5,
               logstd_max: float = 2,
               ) -> None:
    super().__init__()

    self.obs_space = obs_space
    self.action_space = action_space
    self.fft_size = obs_space['spectrum'].shape[0]

    self.hidden_dim = hidden_dim
    self.lr = lr
    self.action_scale = action_scale
    self.action_bias = action_bias
    self.logstd_min = logstd_min
    self.logstd_max = logstd_max
    self.encoder_dim = encoder_dim
    self.feature_dim = feature_dim

    self.features = nn.Sequential(
        nn.Linear(encoder_dim, feature_dim),
        nn.ReLU(),
        nn.LayerNorm(feature_dim),
    )

    self.action_head = nn.Sequential(
        nn.Linear(feature_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, 2*action_space.shape[0])
    )

    self.optimizer = optim.Adam(self.parameters(), lr=lr)
    self.apply(weight_init)

  def forward(self, obs: torch.Tensor) -> torch.Tensor:
    obs = self.features(obs)
    mean, logstd = self.action_head(obs).chunk(2, dim=-1)
    logstd = torch.tanh(logstd)
    logstd = self.logstd_min + 0.5 * \
        (self.logstd_max - self.logstd_min) * (logstd + 1)
    return mean, logstd

  def get_action(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
    mean, logstd = self.forward(x)
    std = logstd.exp()
    action, logprob, mean = squashed_gaussian_action(
        mean, std, self.action_scale, self.action_bias)
    return action, logprob, mean

  def update(self,
             critic: nn.Module,
             obs: torch.Tensor,
             actions: torch.Tensor,
             logprobs: torch.Tensor,
             alpha: float) -> dict:
    q1, q2 = critic(obs, actions)
    min_q = torch.min(q1, q2).view(-1)
    actor_loss = (alpha * logprobs - min_q).mean()

    self.optimizer.zero_grad()
    actor_loss.backward()
    self.optimizer.step()

    info = {
        "losses/actor_loss": actor_loss.item(),
    }

    return info


class Critic(nn.Module):
  def __init__(self,
               obs_space: gym.Space,
               action_space: gym.Space,
               hidden_dim: int,
               lr: float,
               feature_dim: int = 50,
               gamma: float = 0.99,
               ) -> None:
    super().__init__()
    self.obs_space = obs_space
    self.image_shape = obs_space['spectrogram'].shape
    self.fft_size = obs_space['spectrum'].shape[0]
    self.action_shape = action_space
    self.hidden_dim = hidden_dim
    self.lr = lr
    self.gamma = gamma

    self.encoder = ImpalaEncoder(*self.image_shape)
    self.encoder_dim = np.prod(get_out_shape(self.encoder, self.image_shape))

    self.features = nn.Sequential(
        nn.Linear(self.encoder_dim, feature_dim),
        nn.ReLU(),
        nn.LayerNorm(feature_dim),
    )

    self.Q1 = nn.Sequential(
        nn.Linear(feature_dim + action_space[0], hidden_dim),
        nn.ReLU(inplace=True),
        # nn.Linear(hidden_dim, hidden_dim),
        # nn.ReLU(inplace=True),
        nn.Linear(hidden_dim, 1),
    )

    self.Q2 = nn.Sequential(
        nn.Linear(feature_dim + action_space[0], hidden_dim),
        nn.ReLU(inplace=True),
        # nn.Linear(hidden_dim, hidden_dim),
        # nn.ReLU(inplace=True),
        nn.Linear(hidden_dim, 1),
    )

    self.optimizer = optim.Adam(self.parameters(), lr=lr)
    self.encoder_optimizer = optim.Adam(self.encoder.parameters(), lr=lr)
    self.apply(weight_init)

  def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
    assert obs.size(0) == action.size(0)
    obs = self.features(obs)

    obs_action = torch.cat([obs, action], dim=-1)
    q1 = self.Q1(obs_action)
    q2 = self.Q2(obs_action)

    return q1, q2

  def update(self,
             target: nn.Module,
             obs: torch.Tensor,
             next_obs: torch.Tensor,
             actions: torch.Tensor,
             next_actions: torch.Tensor,
             next_logprobs: torch.Tensor,
             rewards: torch.Tensor,
             dones: torch.Tensor,
             alpha: torch.Tensor) -> dict:
    with torch.no_grad():
      q1_next_target, q2_next_target = target(next_obs, next_actions)
      min_q_next_target = torch.min(
          q1_next_target, q2_next_target) - alpha * next_logprobs
      next_q_value = rewards.flatten() + (1 - dones.flatten()) * \
          self.gamma * min_q_next_target.view(-1)

    q1_action_values, q2_action_values = self(obs, actions)
    q1_loss = F.mse_loss(q1_action_values.view(-1), next_q_value)
    q2_loss = F.mse_loss(q2_action_values.view(-1), next_q_value)
    q_loss = q1_loss + q2_loss

    self.optimizer.zero_grad()
    self.encoder_optimizer.zero_grad()
    q_loss.backward()
    self.optimizer.step()
    self.encoder_optimizer.step()

    info = {
        "losses/q1_loss": q1_loss.item(),
        "losses/q2_loss": q2_loss.item(),
        "losses/q1_values": q1_action_values.mean().item(),
        "losses/q2_values": q2_action_values.mean().item(),
    }
    return info

  def update_target_networks(self, target: nn.Module, tau: nn.Module):
    for param, target_param in zip(self.Q1.parameters(),
                                   target.Q1.parameters()):
      target_param.data.copy_(tau * param.data +
                              (1 - tau) * target_param.data)
    for param, target_param in zip(self.Q2.parameters(),
                                   target.Q2.parameters()):
      target_param.data.copy_(tau * param.data +
                              (1 - tau) * target_param.data)


def make_env(env_id, seed, idx, capture_video, run_name, render):
  def thunk():
    if render:
      render_mode = 'human'
    else:
      render_mode = 'rgb_array'
    env = gym.make(env_id,
                   filename='/home/shane/data/HOCAE_Snaps_bool.dat',
                   channel_bandwidth=100e6,
                   fft_size=1024,
                   n_image_snapshots=500,
                   render_mode=render_mode)
    if capture_video:
      if idx == 0:
        env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
    env = gym.wrappers.TimeLimit(env, max_episode_steps=1000)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    # TODO: Allow for frame stacking
    # env = gym.wrappers.FrameStack(env, 4)
    # TODO: Test without reward normalization wrappers
    # env = gym.wrappers.NormalizeReward(env)
    # env = gym.wrappers.TransformReward(env, lambda obs: np.clip(obs, -10, 10))

    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    return env

  return thunk


def weight_init(m):
  """Custom weight init for Conv2D and Linear layers."""
  if isinstance(m, nn.Linear):
    nn.init.orthogonal_(m.weight.data)
    if hasattr(m.bias, 'data'):
      m.bias.data.fill_(0.0)
  elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
    gain = nn.init.calculate_gain('relu')
    nn.init.orthogonal_(m.weight.data, gain)
    if hasattr(m.bias, 'data'):
      m.bias.data.fill_(0.0)


def parse_args():
  # fmt: off
  parser = argparse.ArgumentParser()
  parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
      help="the name of this experiment")
  parser.add_argument("--seed", type=int, default=1,
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

  # Algorithm specific arguments
  parser.add_argument("--env-id", type=str, default="mpar_sim/SpectrumHopperRecorded-v0",
      help="the id of the environment")
  parser.add_argument("--total-timesteps", type=int, default=1000000,
      help="total timesteps of the experiments")
  parser.add_argument("--buffer-size", type=int, default=int(1e6),
      help="the replay memory buffer size")
  parser.add_argument("--gamma", type=float, default=0.99,
      help="the discount factor gamma")
  parser.add_argument("--tau", type=float, default=0.005,
      help="target smoothing coefficient (default: 0.005)")
  parser.add_argument("--batch-size", type=int, default=512,
      help="the batch size of sample from the reply memory")
  parser.add_argument("--exploration-noise", type=float, default=0.1,
      help="the scale of the exploration noise")
  parser.add_argument("--learning_starts", type=int, default=1e3,
      help="timestep to start learning")
  parser.add_argument("--policy-lr", type=float, default=3e-4,
      help="the learning rate of the policy network optimizer")
  parser.add_argument("--q-lr", type=float, default=3e-4,
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
      [make_env(args.env_id, args.seed, 0, args.capture_video, run_name, True)])
  assert isinstance(envs.single_action_space,
                    gym.spaces.Box), "only continuous action space is supported"

  action_scale = torch.Tensor(
      0.5 * (envs.single_action_space.high - envs.single_action_space.low)).to(device)
  action_bias = torch.Tensor(
      0.5 * (envs.single_action_space.high + envs.single_action_space.low)).to(device)

  critic = Critic(obs_space=envs.single_observation_space,
                  action_space=envs.single_action_space.shape,
                  lr=args.q_lr,
                  hidden_dim=256,
                  feature_dim=50,
                  gamma=args.gamma
                  ).to(device)
  critic_target = copy.deepcopy(critic).to(device)
  actor = Actor(obs_space=envs.single_observation_space,
                action_space=envs.single_action_space,
                hidden_dim=256,
                encoder_dim=critic.encoder_dim,
                feature_dim=50,
                lr=args.policy_lr,
                action_scale=action_scale,
                action_bias=action_bias,
                ).to(device)

  # Automatic entropy tuning
  if args.autotune:
    target_entropy = - \
        torch.prod(torch.Tensor(
            envs.single_action_space.shape).to(device)).item()
    log_alpha = torch.zeros(1, requires_grad=True, device=device)
    alpha = log_alpha.exp().item()
    alpha_optimizer = optim.Adam([log_alpha], lr=args.q_lr)
  else:
    alpha = args.alpha

  # Create the replay buffer
  rb = ReplayBuffer(args.buffer_size)
  # Current obs
  rb.create_buffer('spectrograms',
                   envs.single_observation_space['spectrogram'].shape,
                   envs.single_observation_space['spectrogram'].dtype)
  rb.create_buffer('spectrums',
                   envs.single_observation_space['spectrum'].shape,
                   envs.single_observation_space['spectrum'].dtype)
  # Next obs
  rb.create_buffer('next_spectrograms',
                   envs.single_observation_space['spectrogram'].shape,
                   envs.single_observation_space['spectrogram'].dtype)
  rb.create_buffer('next_spectrums',
                   envs.single_observation_space['spectrum'].shape,
                   envs.single_observation_space['spectrum'].dtype)
  rb.create_buffer('actions', envs.single_action_space.shape,
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
      with torch.no_grad():
        encoded_features = critic.encoder(
            torch.Tensor(obs['spectrogram']).to(device))
        # spectrum = torch.Tensor(obs['spectrum']).to(device)
        # encoded_obs = torch.cat([encoded_features, spectrum], dim=-1)
        encoded_obs = encoded_features
        actions, _, _ = actor.get_action(encoded_obs)
        actions = actions.cpu().numpy()

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

    rb.add(spectrograms=obs['spectrogram'],
           spectrums=obs['spectrum'],
           next_spectrograms=real_next_obs['spectrogram'],
           next_spectrums=real_next_obs['spectrum'],
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
      # Observations (more complicated due to the dict space)
      spectrograms = torch.Tensor(data.spectrograms).to(device)
      # spectrums = torch.Tensor(data.spectrums).to(device)
      next_spectrograms = torch.Tensor(data.next_spectrograms).to(device)
      # next_spectrums = torch.Tensor(data.next_spectrums).to(device)
      actions = torch.Tensor(data.actions).to(device)
      rewards = torch.Tensor(data.rewards).to(device)
      dones = torch.Tensor(data.dones).to(device)

      # Encode the observations. We save some computation by only doing the encoding once here.
      encoded_features = critic.encoder(spectrograms)
      with torch.no_grad():
        next_encoded_features = critic.encoder(next_spectrograms)
      # Concatenate before passing to SAC networks
      encoded_obs = encoded_features
      next_encoded_obs = next_encoded_features
      # encoded_obs = torch.cat((encoded_features, spectrums), dim=1)
      # next_encoded_obs = torch.cat(
      #     (next_encoded_features, next_spectrums), dim=1)

      # Train critic
      with torch.no_grad():
        next_actions, next_logprobs, _ = actor.get_action(encoded_obs)
      critic_info = critic.update(target=critic_target,
                                  obs=encoded_obs,
                                  next_obs=next_encoded_obs,
                                  actions=actions,
                                  next_actions=next_actions,
                                  next_logprobs=next_logprobs,
                                  rewards=rewards,
                                  dones=dones,
                                  alpha=alpha)
      train_info.update(critic_info)

      # Train actor
      if global_step % args.policy_frequency == 0:
        for _ in range(args.policy_frequency):
          a, logprob_a, _ = actor.get_action(encoded_obs.detach())
          actor_info = actor.update(
              critic=critic,
              obs=encoded_obs.detach(),
              actions=a,
              logprobs=logprob_a,
              alpha=alpha)
          train_info.update(actor_info)

          if args.autotune:
            with torch.no_grad():
              _, logprob, _ = actor.get_action(encoded_obs)
            alpha_loss = (-log_alpha * (logprob + target_entropy)).mean()

            alpha_optimizer.zero_grad()
            alpha_loss.backward()
            alpha_optimizer.step()
            alpha = log_alpha.exp().item()

      # update the target networks
      if global_step % args.target_network_frequency == 0:
        critic.update_target_networks(critic_target, tau=args.tau)

      # Write statistics to tensorboard
      if global_step % 100 == 0:
        for key, value in train_info.items():
          writer.add_scalar(key, value, global_step)
        writer.add_scalar("losses/alpha", alpha, global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step /
                          (time.time() - start_time)), global_step)
        if args.autotune:
          writer.add_scalar("losses/alpha_loss",
                            alpha_loss.item(), global_step)
