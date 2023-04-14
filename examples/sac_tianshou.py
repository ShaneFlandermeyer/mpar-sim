# %%
import datetime
import os
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Independent, Normal
from torch.utils.tensorboard import SummaryWriter

from tianshou.data import Collector, VectorReplayBuffer
from tianshou.env import SubprocVectorEnv, ShmemVectorEnv
from tianshou.policy import PPOPolicy, SACPolicy
from tianshou.trainer import onpolicy_trainer, offpolicy_trainer
from tianshou.utils.net.common import ActorCritic, Net
from tianshou.utils.net.continuous import ActorProb, Critic
from tianshou.utils import TensorboardLogger
from tianshou.data import Collector, ReplayBuffer, VectorReplayBuffer

import mpar_sim.envs

import warnings
warnings.filterwarnings('ignore')

# %%


def make_env(env_id, idx):
  env = gym.make(env_id,
                 filename='/home/shane/data/HOCAE_Snaps_bool_cleaned.dat',
                 channel_bandwidth=100e6,
                 fft_size=1024,
                 render_mode="rgb_array")
  env = gym.wrappers.TimeLimit(env, max_episode_steps=250)
  return env


# %%
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# environments
env_id = "mpar_sim/SpectrumHopper1D-v0"
env = gym.make(env_id,
               filename='/home/shane/data/HOCAE_Snaps_bool_cleaned.dat',
               channel_bandwidth=100e6,
               fft_size=1024,
               )
state_shape = env.observation_space.shape or env.observation_space.n
action_shape = env.action_space.shape or env.action_space.n
max_action = env.action_space.high[0]
print("Observations shape:", state_shape)
print("Actions shape:", action_shape)
print("Action range:", np.min(env.action_space.low),
      np.max(env.action_space.high))
train_envs = ShmemVectorEnv([lambda: make_env(env_id, i) for i in range(1)])
test_envs = ShmemVectorEnv([lambda: make_env(env_id, i) for i in range(10)])

# model & optimizer
actor_net = Net(
    env.observation_space.shape,
    hidden_sizes=[256, 256],
    device=device)
critic_net1 = Net(
    env.observation_space.shape,
    env.action_space.shape,
    hidden_sizes=[256, 256],
    concat=True,
    device=device)
critic_net2 = Net(
    env.observation_space.shape,
    env.action_space.shape,
    hidden_sizes=[256, 256],
    concat=True,
    device=device)
actor = ActorProb(
    actor_net,
    env.action_space.shape,
    max_action=env.action_space.high[0],
    unbounded=True,
    device=device,
    conditioned_sigma=True).to(device)
critic1 = Critic(critic_net1, device=device).to(device)
critic2 = Critic(critic_net2, device=device).to(device)

critic1_optim = torch.optim.Adam(critic1.parameters(), lr=1e-3)
critic2_optim = torch.optim.Adam(critic2.parameters(), lr=1e-3)
actor_optim = torch.optim.Adam(actor.parameters(), lr=1e-3)

target_entropy = -np.prod(env.action_space.shape)
log_alpha = torch.zeros(1, requires_grad=True, device=device)
alpha_optim = torch.optim.Adam([log_alpha], lr=3e-4)
alpha = (target_entropy, log_alpha, alpha_optim)

# PPO policy
policy = SACPolicy(
    actor,
    actor_optim,
    critic1,
    critic1_optim,
    critic2,
    critic2_optim,
    tau=0.005,
    gamma=0.99,
    alpha=alpha,
    estimation_step=1,
    action_space=env.action_space,
)


# collector
buffer = ReplayBuffer(1_000_000)
train_collector = Collector(policy, train_envs, buffer, exploration_noise=True)
test_collector = Collector(policy, test_envs)
train_collector.collect(n_step=10000, random=True)

# log
logdir = "log"
now = datetime.datetime.now().strftime("%y%m%d-%H%M%S")
algo_name = "ppo"
log_name = os.path.join("spectrum_hopper", algo_name, now)
log_path = os.path.join(logdir, log_name)
writer = SummaryWriter(log_path)
logger = TensorboardLogger(writer)

# trainer
result = offpolicy_trainer(
    policy,
    train_collector,
    test_collector,
    200,
    5000,
    1,
    10,
    256,
    logger=logger,
    update_per_step=1,
    test_in_train=False,
)
print(result)

# %% [markdown]
#
