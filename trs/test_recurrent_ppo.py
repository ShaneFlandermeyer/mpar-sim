# %%
hp = None
ENV = None
ENV_MASK_VELOCITY = None
WORKSPACE_PATH = "/home/shane/src/mpar-sim/trs"



# %%
import torch
from torch import nn
from torch import distributions
import gymnasium as gym
import copy
from torch.utils.tensorboard import SummaryWriter
from torch import optim
import numpy as np
from itertools import count
import torch.nn.functional as F
import re
import copy
import time
import math
import pathlib
import time
import pickle
import os
from dataclasses import dataclass
import gc
from dotmap import DotMap
from base64 import b64encode
from IPython.display import HTML
from typing import Tuple
import mpar_sim.envs

# %%

def layer_init(layer: nn.Module, std=np.sqrt(2), bias_const=0.0) -> nn.Module:
  torch.nn.init.orthogonal_(layer.weight, std)
  torch.nn.init.constant_(layer.bias, bias_const)
  return layer


class PositionalEncoding(nn.Module):

  def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
    super().__init__()
    self.dropout = nn.Dropout(p=dropout)

    position = torch.arange(max_len).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2) *
                         (-math.log(10000.0) / d_model))
    pe = torch.zeros(1, max_len, d_model)
    pe[0, :, 0::2] = torch.sin(position * div_term)
    pe[0, :, 1::2] = torch.cos(position * div_term)
    self.register_buffer('pe', pe)

  def forward(self, x: torch.tensor) -> torch.Tensor:
    """
    Arguments:
        x: Tensor, shape ``[batch_size, seq_len, embedding_dim]``
    """
    inp = x
    x = x + self.pe[:, :x.size(-2)]
    assert x.shape == inp.shape
    return self.dropout(x)


class Actor(nn.Module):
  def __init__(self, 
               obs_shape: Tuple[int], 
               action_dim: int, 
               continuous_action_space: bool):
    super().__init__()
    self.obs_shape = obs_shape
    self.action_dim = action_dim
    self.n_embed = hp.hidden_size
    self.num_recurrent_layers = hp.recurrent_layers
    self.position_encoding = PositionalEncoding(self.n_embed, 
                                                dropout=0, max_len=10_000)

    self.embed = layer_init(nn.Linear(self.obs_shape[1], self.n_embed))
    self.mha = nn.MultiheadAttention(self.n_embed, 1, batch_first=True)
    self.layernorm = nn.LayerNorm(self.n_embed)
    self.lstm = nn.LSTM(input_size=self.n_embed, 
                        hidden_size=self.n_embed,
                        num_layers=hp.recurrent_layers)
    self.out = layer_init(nn.Linear(self.n_embed, 2*action_dim), std=0.01)

    self.continuous_action_space = continuous_action_space
    self.hidden_cell = None

  def get_init_state(self, batch_size: int, device: torch.device):
    self.hidden_cell = (torch.zeros(hp.recurrent_layers, batch_size, hp.hidden_size).to(device),
                        torch.zeros(hp.recurrent_layers, batch_size, hp.hidden_size).to(device))

  def forward(self, state: torch.tensor, terminal: torch.tensor = None):
    seq_len, batch_size = state.shape[:2]
    state = state.reshape(seq_len, batch_size, *self.obs_shape)
    # Sub-PRI attention processing
    x = self.embed(state)
    x = F.elu(x).reshape(-1, *x.shape[-2:])
    x = self.position_encoding(x)
    x, _ = self.mha(x, x, x, need_weights=False)
    x = F.elu(x).reshape(seq_len, batch_size, *x.shape[-2:])
    x = x.mean(dim=-2)
    mha_out = x.reshape(seq_len, batch_size, -1)
    

    # Pulse-to-pulse recurrent processing
    device = state.device
    if self.hidden_cell is None or batch_size != self.hidden_cell[0].shape[1]:
      self.get_init_state(batch_size, device)
    if terminal is not None:
      self.hidden_cell = [
          value * (1.0 - terminal).reshape(1, batch_size, 1) for value in self.hidden_cell]
    _, self.hidden_cell = self.lstm(x, self.hidden_cell)
    # Concatenate attenttion out and recurrent out
    # TODO: Concat instead of add
    x = F.elu(mha_out[-1] + self.hidden_cell[0][-1]) 
    # x = torch.cat((mha_out[-1], self.hidden_cell[0][-1]), dim=-1)
    policy_logits = self.out(x)
    
    # Convert to action distribution
    if self.continuous_action_space:
      mu, sigma = torch.chunk(policy_logits, 2, dim=-1)
      sigma = nn.functional.softplus(sigma)
      policy_dist = torch.distributions.multivariate_normal.MultivariateNormal(
          mu.to("cpu"), torch.diag_embed(sigma).to("cpu"))
    else:
      policy_dist = distributions.Categorical(
          F.softmax(policy_logits, dim=1).to("cpu"))
    return policy_dist


class Critic(nn.Module):
  def __init__(self, obs_shape):
    super().__init__()
    self.obs_shape = obs_shape
    self.n_embed = hp.hidden_size
    self.position_encoding = PositionalEncoding(self.n_embed)

    self.embed = layer_init(nn.Linear(self.obs_shape[1], self.n_embed))
    self.mha = nn.MultiheadAttention(self.n_embed, 1, batch_first=True)
    self.layernorm = nn.LayerNorm(self.n_embed)
    

    self.lstm = nn.LSTM(self.n_embed, 
                        self.n_embed, 
                        num_layers=hp.recurrent_layers)
    self.out = layer_init(nn.Linear(self.n_embed, 1), std=1)
    
    self.hidden_cell = None

  def get_init_state(self, batch_size, device):
    self.hidden_cell = (torch.zeros(hp.recurrent_layers, batch_size, hp.hidden_size).to(device),
                        torch.zeros(hp.recurrent_layers, batch_size, hp.hidden_size).to(device))

  def forward(self, state: torch.tensor, terminal: torch.tensor = None):
    inp = state # TODO: Needed for debugging only
    seq_len, batch_size = state.shape[:2]
    state = state.reshape(seq_len, batch_size, *self.obs_shape)
    
    # Sub-PRI attention processing
    x = self.embed(state)
    x = F.elu(x).reshape(-1, *x.shape[-2:])
    x = self.position_encoding(x)
    x, _ = self.mha(x, x, x, need_weights=False)
    x = F.elu(x).reshape(seq_len, batch_size, *x.shape[-2:])
    x = x.mean(dim=-2)
    mha_out = x.reshape(seq_len, batch_size, -1)
    
    # Pulse-to-pulse recurrent processing
    device = state.device
    if self.hidden_cell is None or batch_size != self.hidden_cell[0].shape[1]:
      self.get_init_state(batch_size, device)
    if terminal is not None:
      self.hidden_cell = [
          value * (1.0 - terminal).reshape(1, batch_size, 1) for value in self.hidden_cell]
    _, self.hidden_cell = self.lstm(x, self.hidden_cell)
    # TODO: Concat instead of add
    x = F.elu(mha_out[-1] + self.hidden_cell[0][-1]) 
    # x = torch.cat((mha_out[-1], self.hidden_cell[0][-1]), dim=-1)
    
    value_out = self.out(x)
    return value_out


# %%
@dataclass
class HyperParameters():
    scale_reward:         float
    min_reward:           float
    hidden_size:          float
    batch_size:           int
    discount:             float
    gae_lambda:           float
    ppo_clip:             float
    ppo_epochs:           int
    max_grad_norm:        float
    entropy_factor:       float
    actor_learning_rate:  float
    critic_learning_rate: float
    recurrent_seq_len:    int
    recurrent_layers:     int
    rollout_steps:        int
    parallel_rollouts:    int
    patience:             int
    # Apply to continous action spaces only 
    # trainable_std_dev:    bool
    # init_log_std_dev:     float
@dataclass
class StopConditions():
    """
    Store parameters and variables used to stop training. 
    """
    best_reward: float = -1e6
    fail_to_improve_count: int = 0
    max_iterations: int = 1000000
    
def get_env_space():
  """
  Return obsvervation dimensions, action dimensions and whether or not action space is continuous.
  """
  env = gym.make(ENV)
  # NOTE: This should in general be done with the assignment below, but this env is a little weird.
  obs_shape = env.observation_space.shape
  continuous_action_space = type(env.action_space) is gym.spaces.box.Box
  if continuous_action_space:
    action_dim = env.action_space.shape[0]
  else:
    action_dim = env.action_space.n
#   obs_shape = env.observation_space.shape
  return obs_shape, action_dim, continuous_action_space

def load_checkpoint(iteration):
    """
    Load from training checkpoint.
    """
    global ENV
    global ENV_MASK_VELOCITY
    global hp
    CHECKPOINT_PATH = BASE_CHECKPOINT_PATH + f"{iteration}/"
    with open(CHECKPOINT_PATH + "parameters.pt", "rb") as f:
        checkpoint = pickle.load(f)
        
    ENV = checkpoint.env
    ENV_MASK_VELOCITY = checkpoint.env_mask_velocity
    hp = checkpoint.hp

    actor_state_dict = torch.load(CHECKPOINT_PATH + "actor.pt", map_location=torch.device("cpu"))
    critic_state_dict = torch.load(CHECKPOINT_PATH + "critic.pt", map_location=torch.device("cpu"))
    actor_optimizer_state_dict = torch.load(CHECKPOINT_PATH + "actor_optimizer.pt", map_location=torch.device("cpu"))
    critic_optimizer_state_dict = torch.load(CHECKPOINT_PATH + "critic_optimizer.pt", map_location=torch.device("cpu"))
    
    return (actor_state_dict, critic_state_dict,
           actor_optimizer_state_dict, critic_optimizer_state_dict,
           checkpoint.stop_conditions)

def load_from_checkpoint(max_checkpoint_iteration):
    
    actor_state_dict, critic_state_dict, actor_optimizer_state_dict, critic_optimizer_state_dict, stop_conditions = load_checkpoint(max_checkpoint_iteration)
    
    obsv_dim, action_dim, continuous_action_space = get_env_space()
    actor = Actor(obsv_dim,
                  action_dim,
                  continuous_action_space=continuous_action_space)
    critic = Critic(obsv_dim)
    
    actor_optimizer = optim.AdamW(actor.parameters(), lr=hp.actor_learning_rate)
    critic_optimizer = optim.AdamW(critic.parameters(), lr=hp.critic_learning_rate)

    actor.load_state_dict(actor_state_dict, strict=True) 
    critic.load_state_dict(critic_state_dict, strict=True)
    actor_optimizer.load_state_dict(actor_optimizer_state_dict)
    critic_optimizer.load_state_dict(critic_optimizer_state_dict)

    # We have to move manually move optimizer states to TRAIN_DEVICE manually since optimizer doesn't yet have a "to" method.
    for state in actor_optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to("cpu")

    for state in critic_optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to("cpu")

    return actor, critic, actor_optimizer, critic_optimizer, max_checkpoint_iteration, stop_conditions


def visualise_policy(actor):
    """
    Visualise policy.
    """
    env = gym.make(ENV, render_mode='human')
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env = gym.wrappers.FlattenObservation(env)
    env = gym.wrappers.ClipAction(env)
    # env.render_mode = "human"
    # env = gym.wrappers.Monitor(env, f"{WORKSPACE_PATH}/videos", force=True)
    observation, _ = env.reset()
    done_mask = torch.zeros(1) 
    done = False
    step_idx = 0
    actor = actor.to("cpu")
    actor.eval()
    actor.get_init_state(1, "cpu")
    total_reward = 0.
    print("Testing policy...")
    while(not done):
        # Choose next action 
        state = torch.tensor(observation, dtype=torch.float32)
        dist = actor(state.reshape([1, 1, -1]), done_mask)
        action =  dist.sample().squeeze(0)
        # Apply action
        action_np = action.cpu().numpy()
        observation, reward, term, trunc, info = env.step(action_np)
        env.render()
        done = term or trunc
        step_idx += 1
        total_reward += reward
    # print(total_reward)
    print(f"Steps to done: {step_idx}, Total reward: {total_reward}")
    return

class MaskVelocityWrapper(gym.ObservationWrapper):
    """
    Gym environment observation wrapper used to mask velocity terms in
    observations. The intention is the make the MDP partially observatiable.
    """
    def __init__(self, env):
        super(MaskVelocityWrapper, self).__init__(env)
        if ENV == "CartPole-v1":
            self.mask = np.array([1., 0., 1., 0.])
        elif ENV == "Pendulum-v0":
            self.mask = np.array([1., 1., 0.])
        elif ENV == "LunarLander-v2":
            self.mask = np.array([1., 1., 0., 0., 1., 0., 1., 1,])
        elif ENV == "LunarLanderContinuous-v2":
            self.mask = np.array([1., 1., 0., 0., 1., 0., 1., 1,])
        else:
            raise NotImplementedError

    def observation(self, observation):
        return  observation * self.mask

# %%
EXPERIMENT_NAME = "SpectrumEnv_20231019_170531"
BASE_CHECKPOINT_PATH = f"{WORKSPACE_PATH}/checkpoints/{EXPERIMENT_NAME}/"

# %%
actor, critic, actor_optimizer, critic_optimizer, iteration, stop_conditions = load_from_checkpoint(220)



# %%
file_infix = visualise_policy(actor=actor)

# %%
# mp4 = open(f"{WORKSPACE_PATH}/videos/openaigym.video.{file_infix}.video000000.mp4",'rb').read()
# data_url = "data:video/mp4;base64," + b64encode(mp4).decode()
# HTML("""
# <video width=400 controls>
#       <source src="%s" type="video/mp4">
# </video>
# """ % data_url)

# %%


# %%


# %%



