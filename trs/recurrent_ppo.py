from typing import Dict, Tuple
from IPython.display import HTML
from base64 import b64encode
from dotmap import DotMap
import gc
from dataclasses import dataclass
import os
import pickle
import pathlib
import math
import time
import re
import torch.nn.functional as F
from itertools import count
import numpy as np
from torch import optim
from torch.utils.tensorboard import SummaryWriter
import copy
import gymnasium as gym
from torch import distributions
from torch import nn
import torch
from mpar_sim.wrappers.squash_action import SquashAction
from mpar_sim.wrappers.first_n import TakeFirstN
from mpar_sim.wrappers.add_pulse_index import AddPulseIndex

WORKSPACE_PATH = "/home/shane/src/mpar-sim/trs"


# Select version 2 of tensorflow to avoid warnings.
# %tensorflow_version 2.x

# %% [markdown]
# .

# %% [markdown]
# # Settings

# %%

# Save metrics for viewing with tensorboard.
SAVE_METRICS_TENSORBOARD = True

# Save actor & critic parameters for viewing in tensorboard.
SAVE_PARAMETERS_TENSORBOARD = False

# Save training state frequency in PPO iterations.
CHECKPOINT_FREQUENCY = 10

# Step env asynchronously using multiprocess or synchronously.
ASYNCHRONOUS_ENVIRONMENT = True

# Force using CPU for gathering trajectories.
FORCE_CPU_GATHER = False

RANDOM_SEED = np.random.randint(0, 2**32 - 1)
# RANDOM_SEED = 1
# Set random seed for consistant runs.
torch.random.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
# Set maximum threads for torch to avoid inefficient use of multiple cpu cores.
torch.set_num_threads(1)
TRAIN_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
GATHER_DEVICE = "cuda" if torch.cuda.is_available() and not FORCE_CPU_GATHER else "cpu"

# %% [markdown]
# # Parameters

# %%
# Environment parameters
ENV = "mpar_sim/SpectrumEnv"
adapt_weight = 0
collision_weight = 30*(1.0 - adapt_weight)
print(f"w = {adapt_weight}, b = {collision_weight}")
ENV_KWARGS = dict(
    dataset="/home/shane/data/hocae_snaps_2_4_cleaned_10_0.dat",
    # dataset="/home/shane/data/hocae_snaps_2_64_cleaned_10_0.dat",
    pri=10,
    order="C",
    collision_weight=collision_weight,
    adapt_weight_bw=adapt_weight,
    adapt_weight_fc=adapt_weight,
)
TAKE_FIRST_N = 10
EXPERIMENT_NAME = f"2_4/gamma=0.9/adapt={adapt_weight}/SpectrumEnv" + \
    f"_{time.strftime('%Y%m%d_%H%M%S')}" + f"_{RANDOM_SEED}"

# Default Hyperparameters
SCALE_REWARD:         float = 1
HIDDEN_SIZE:          float = 64
BATCH_SIZE:           int = 256
DISCOUNT:             float = 0.7
GAE_LAMBDA:           float = 0.95
PPO_CLIP:             float = 0.2
PPO_EPOCHS:           int = 10
MAX_GRAD_NORM:        float = 1.
ENTROPY_FACTOR:       float = 0
ACTOR_LEARNING_RATE:  float = 2e-4
CRITIC_LEARNING_RATE: float = 2e-4
RECURRENT_SEQ_LEN:    int = 4
RECURRENT_LAYERS:     int = 1
ROLLOUT_STEPS:        int = 512
PARALLEL_ROLLOUTS:    int = 16
PATIENCE:             int = np.Inf

# %% [markdown]
# # Hyperparameters

# %%


@dataclass
class HyperParameters():
  scale_reward:         float = SCALE_REWARD
  hidden_size:          float = HIDDEN_SIZE
  batch_size:           int = BATCH_SIZE
  discount:             float = DISCOUNT
  gae_lambda:           float = GAE_LAMBDA
  ppo_clip:             float = PPO_CLIP
  ppo_epochs:           int = PPO_EPOCHS
  max_grad_norm:        float = MAX_GRAD_NORM
  entropy_factor:       float = ENTROPY_FACTOR
  actor_learning_rate:  float = ACTOR_LEARNING_RATE
  critic_learning_rate: float = CRITIC_LEARNING_RATE
  recurrent_seq_len:    int = RECURRENT_SEQ_LEN
  recurrent_layers:     int = RECURRENT_LAYERS
  rollout_steps:        int = ROLLOUT_STEPS
  parallel_rollouts:    int = PARALLEL_ROLLOUTS
  patience:             int = PATIENCE


hp = HyperParameters(batch_size=BATCH_SIZE, parallel_rollouts=PARALLEL_ROLLOUTS, recurrent_seq_len=RECURRENT_SEQ_LEN, rollout_steps=ROLLOUT_STEPS, patience=PATIENCE, entropy_factor=ENTROPY_FACTOR,
                     hidden_size=HIDDEN_SIZE)


# %%
num_minibatch = hp.parallel_rollouts * \
    hp.rollout_steps / hp.recurrent_seq_len / hp.batch_size
print(f"num_minibatch: {num_minibatch}")
assert num_minibatch >= 1., "Less than 1 batch per trajectory.  Are you sure that's what you want?"

# %%


def calc_discounted_return(rewards, discount, final_value):
  """
  Calculate discounted returns based on rewards and discount factor.
  """
  seq_len = len(rewards)
  discounted_returns = torch.zeros(seq_len)
  discounted_returns[-1] = rewards[-1] + discount * final_value
  for i in range(seq_len - 2, -1, -1):
    discounted_returns[i] = rewards[i] + discount * discounted_returns[i + 1]
  return discounted_returns

# %%


def compute_advantages(rewards, values, discount, gae_lambda):
  """
  Compute General Advantage.
  """
  deltas = rewards + discount * values[1:] - values[:-1]
  seq_len = len(rewards)
  advs = torch.zeros(seq_len + 1)
  multiplier = discount * gae_lambda
  for i in range(seq_len - 1, -1, -1):
    advs[i] = advs[i + 1] * multiplier + deltas[i]
  return advs[:-1]

# %% [markdown]
# # Helper functions


# %%
_INVALID_TAG_CHARACTERS = re.compile(r"[^-/\w\.]")
BASE_CHECKPOINT_PATH = f"{WORKSPACE_PATH}/checkpoints/ppo_continuous/{EXPERIMENT_NAME}/"


def save_parameters(writer, tag, model, batch_idx):
  """
  Save model parameters for tensorboard.
  """
  for k, v in model.state_dict().items():
    shape = v.shape
    # Fix shape definition for tensorboard.
    shape_formatted = _INVALID_TAG_CHARACTERS.sub("_", str(shape))
    # Don't do this for single weights or biases
    if np.any(np.array(shape) > 1):
      mean = torch.mean(v)
      std_dev = torch.std(v)
      maximum = torch.max(v)
      minimum = torch.min(v)
      writer.add_scalars(
          "{}_weights/{}{}".format(tag, k, shape_formatted),
          {"mean": mean, "std_dev": std_dev, "max": maximum, "min": minimum},
          batch_idx,
      )
    else:
      writer.add_scalar("{}_{}{}".format(
          tag, k, shape_formatted), v.data, batch_idx)


def log_metrics(infos: Dict, writer: SummaryWriter, step: int):
  mean_bws = []
  mean_cols = []
  mean_widests = []
  mean_missed_bws = []
  mean_bw_diffs = []
  mean_fc_diffs = []
  for info in infos["final_info"]:
    # Skip the envs that are not done
    if info is None:
      continue

    mean_bws.append(info["mean_bw"])
    mean_cols.append(info["mean_collision_bw"])
    mean_widests.append(info["mean_widest_bw"])
    mean_missed_bws.append(info["mean_missed_bw"])
    mean_bw_diffs.append(info["mean_bw_diff"])
    mean_fc_diffs.append(info["mean_fc_diff"])

  if len(mean_bws) > 0:
    avg_mean_bw = np.mean(mean_bws)
    avg_mean_col = np.mean(mean_cols)
    avg_mean_widest = np.mean(mean_widests)
    # avg_mean_missed_bw = abs(avg_mean_bw - avg_mean_widest)
    avg_mean_missed_bw = np.mean(mean_missed_bws)
    avg_mean_bw_diff = np.mean(mean_bw_diffs)
    avg_mean_fc_diff = np.mean(mean_fc_diffs)
    writer.add_scalar("charts/mean_bandwidth", avg_mean_bw, step)
    writer.add_scalar("charts/mean_collision_bw", avg_mean_col, step)
    writer.add_scalar("charts/mean_widest_bw", avg_mean_widest, step)
    writer.add_scalar("charts/mean_missed_bw", avg_mean_missed_bw, step)
    writer.add_scalar("charts/mean_bw_diff", avg_mean_bw_diff, step)
    writer.add_scalar("charts/mean_fc_diff", avg_mean_fc_diff, step)
    writer.add_scalar("charts/episodic_return",
                      infos['final_info'][0]["episode"]["r"], step)
    writer.add_scalar("charts/episodic_length",
                      infos['final_info'][0]["episode"]["l"], step)
    print(
        f"global_step={step}, bw={avg_mean_bw:.3f}, col={avg_mean_col:.3f}, widest={avg_mean_widest:.3f}")


def get_env_space():
  """
  Return obsvervation dimensions, action dimensions and whether or not action space is continuous.
  """
  env = gym.make(ENV, **ENV_KWARGS)
  # NOTE: This should in general be done with the assignment below, but this env is a little weird.
  obs_shape = env.observation_space.shape
  continuous_action_space = type(env.action_space) is gym.spaces.box.Box
  if continuous_action_space:
    action_dim = env.action_space.shape[0]
  else:
    action_dim = env.action_space.n
#   obs_shape = env.observation_space.shape
  return obs_shape, action_dim, continuous_action_space


def get_last_checkpoint_iteration():
  """
  Determine latest checkpoint iteration.
  """
  if os.path.isdir(BASE_CHECKPOINT_PATH):
    max_checkpoint_iteration = max(
        [int(dirname) for dirname in os.listdir(BASE_CHECKPOINT_PATH)])
  else:
    max_checkpoint_iteration = 0
  return max_checkpoint_iteration


def save_checkpoint(actor, critic, actor_optimizer, critic_optimizer, iteration, stop_conditions):
  """
  Save training checkpoint.
  """
  checkpoint = DotMap()
  checkpoint.env = ENV
  checkpoint.iteration = iteration
  checkpoint.stop_conditions = stop_conditions
  checkpoint.hp = hp
  CHECKPOINT_PATH = BASE_CHECKPOINT_PATH + f"{iteration}/"
  pathlib.Path(CHECKPOINT_PATH).mkdir(parents=True, exist_ok=True)
  with open(CHECKPOINT_PATH + "parameters.pt", "wb") as f:
    pickle.dump(checkpoint, f)
  with open(CHECKPOINT_PATH + "actor_class.pt", "wb") as f:
    pickle.dump(Actor, f)
  with open(CHECKPOINT_PATH + "critic_class.pt", "wb") as f:
    pickle.dump(Critic, f)
  torch.save(actor.state_dict(), CHECKPOINT_PATH + "actor.pt")
  torch.save(critic.state_dict(), CHECKPOINT_PATH + "critic.pt")
  torch.save(actor_optimizer.state_dict(),
             CHECKPOINT_PATH + "actor_optimizer.pt")
  torch.save(critic_optimizer.state_dict(),
             CHECKPOINT_PATH + "critic_optimizer.pt")


def start_or_resume_from_checkpoint():
  """
  Create actor, critic, actor optimizer and critic optimizer from scratch
  or load from latest checkpoint if it exists. 
  """
  max_checkpoint_iteration = get_last_checkpoint_iteration()

  obs_shape, action_dim, continuous_action_space = get_env_space()
  actor = Actor(obs_shape=obs_shape,
                action_dim=action_dim,
                continuous_action_space=continuous_action_space)
  critic = Critic(obs_shape=obs_shape, action_dim=action_dim)

  actor_optimizer = optim.AdamW(actor.parameters(), lr=hp.actor_learning_rate)
  critic_optimizer = optim.AdamW(
      critic.parameters(), lr=hp.critic_learning_rate)

  stop_conditions = StopConditions()

  # If max checkpoint iteration is greater than zero initialise training with the checkpoint.
  if max_checkpoint_iteration > 0:
    actor_state_dict, critic_state_dict, actor_optimizer_state_dict, critic_optimizer_state_dict, stop_conditions = load_checkpoint(
        max_checkpoint_iteration)

    actor.load_state_dict(actor_state_dict, strict=True)
    critic.load_state_dict(critic_state_dict, strict=True)
    actor_optimizer.load_state_dict(actor_optimizer_state_dict)
    critic_optimizer.load_state_dict(critic_optimizer_state_dict)

    # We have to move manually move optimizer states to TRAIN_DEVICE manually since optimizer doesn't yet have a "to" method.
    for state in actor_optimizer.state.values():
      for k, v in state.items():
        if isinstance(v, torch.Tensor):
          state[k] = v.to(TRAIN_DEVICE)

    for state in critic_optimizer.state.values():
      for k, v in state.items():
        if isinstance(v, torch.Tensor):
          state[k] = v.to(TRAIN_DEVICE)

  return actor, critic, actor_optimizer, critic_optimizer, max_checkpoint_iteration, stop_conditions


def load_checkpoint(iteration):
  """
  Load from training checkpoint.
  """
  CHECKPOINT_PATH = BASE_CHECKPOINT_PATH + f"{iteration}/"
  with open(CHECKPOINT_PATH + "parameters.pt", "rb") as f:
    checkpoint = pickle.load(f)

  assert ENV == checkpoint.env, "To resume training environment must match current settings."
  assert hp == checkpoint.hp, "To resume training hyperparameters must match current settings."

  actor_state_dict = torch.load(
      CHECKPOINT_PATH + "actor.pt", map_location=torch.device(TRAIN_DEVICE))
  critic_state_dict = torch.load(
      CHECKPOINT_PATH + "critic.pt", map_location=torch.device(TRAIN_DEVICE))
  actor_optimizer_state_dict = torch.load(
      CHECKPOINT_PATH + "actor_optimizer.pt", map_location=torch.device(TRAIN_DEVICE))
  critic_optimizer_state_dict = torch.load(
      CHECKPOINT_PATH + "critic_optimizer.pt", map_location=torch.device(TRAIN_DEVICE))

  return (actor_state_dict, critic_state_dict,
          actor_optimizer_state_dict, critic_optimizer_state_dict,
          checkpoint.stop_conditions)


@dataclass
class StopConditions():
  """
  Store parameters and variables used to stop training. 
  """
  best_reward: float = -1e6
  fail_to_improve_count: int = 0
  max_iterations: int = 201
# %% [markdown]
# # Recurrent Models

# %%


def layer_init(layer: nn.Module, std=np.sqrt(2), bias_const=0.0) -> nn.Module:
  torch.nn.init.orthogonal_(layer.weight, std)
  torch.nn.init.constant_(layer.bias, bias_const)
  return layer


class PositionalEncoding(nn.Module):

  def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 10_000):
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
               continuous_action_space: bool,):
    super().__init__()
    self.obs_shape = obs_shape
    self.action_dim = action_dim
    self.n_embed = hp.hidden_size
    self.num_recurrent_layers = hp.recurrent_layers
    self.position_encoding = PositionalEncoding(self.n_embed,
                                                dropout=0.1, max_len=10_000)

    self.embed = layer_init(nn.Linear(self.obs_shape[1], self.n_embed))
    self.mha = nn.MultiheadAttention(self.n_embed, 4, batch_first=True)
    self.layernorm = nn.LayerNorm(self.n_embed)
    self.rnn = nn.GRU(input_size=self.n_embed+action_dim+1,
                      hidden_size=self.n_embed,
                      num_layers=hp.recurrent_layers)
    self.out = layer_init(nn.Linear(self.n_embed, 2*action_dim), std=0.01)

    self.continuous_action_space = continuous_action_space
    self.hidden_cell = None

  def get_init_state(self, batch_size: int, device: torch.device):
    self.hidden_cell = torch.zeros(
        hp.recurrent_layers, batch_size, hp.hidden_size).to(device)

  def forward(self,
              state: torch.tensor,
              prev_action: torch.tensor,
              prev_reward: torch.tensor,
              terminal: torch.tensor = None):
    seq_len, batch_size = state.shape[:2]
    # state, pulse_index = state.split(state.shape[-1] - 1, dim=-1)
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
    if self.hidden_cell is None or batch_size != self.hidden_cell.shape[1]:
      self.get_init_state(batch_size, device)
    if terminal is not None:
      self.hidden_cell *= (1.0 - terminal).reshape(1, batch_size, 1)
    # Add previous action to lstm input
    x = torch.cat((x, prev_action, prev_reward), dim=-1)
    _, self.hidden_cell = self.rnn(x, self.hidden_cell)

    # Skip path from MHA to the output
    x = F.elu(mha_out[-1] + self.hidden_cell[-1])
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
  def __init__(self, obs_shape, action_dim):
    super().__init__()
    self.obs_shape = obs_shape
    self.action_dim = action_dim
    self.n_embed = hp.hidden_size
    self.position_encoding = PositionalEncoding(self.n_embed,
                                                dropout=0.1, max_len=10_000)

    self.embed = layer_init(nn.Linear(self.obs_shape[1], self.n_embed))
    self.mha = nn.MultiheadAttention(self.n_embed, 4, batch_first=True)
    self.layernorm = nn.LayerNorm(self.n_embed)

    self.rnn = nn.GRU(self.n_embed+action_dim+1,
                       self.n_embed,
                       num_layers=hp.recurrent_layers)
    self.out = layer_init(nn.Linear(self.n_embed, 1), std=1)

    self.hidden_cell = None

  def get_init_state(self, batch_size, device):
    self.hidden_cell = torch.zeros(
        hp.recurrent_layers, batch_size, hp.hidden_size).to(device)

  def forward(self,
              state: torch.tensor,
              prev_action: torch.tensor,
              prev_reward: torch.tensor,
              terminal: torch.tensor = None):
    seq_len, batch_size = state.shape[:2]
    # state, pulse_index = state.split(state.shape[-1] - 1, dim=-1)
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
    if self.hidden_cell is None or batch_size != self.hidden_cell.shape[1]:
      self.get_init_state(batch_size, device)
    if terminal is not None:
      self.hidden_cell *= (1.0 - terminal).reshape(1, batch_size, 1)
    x = torch.cat((x, prev_action, prev_reward), dim=-1)
    _, self.hidden_cell = self.rnn(x, self.hidden_cell)

    # Skip path from MHA to the output
    x = F.elu(mha_out[-1] + self.hidden_cell[-1])

    value_out = self.out(x)
    return value_out

# %% [markdown]
# # Gather trajectories from environment using current policy.


# %%
global_step = 0


def gather_trajectories(input_data):
  """
  Gather policy trajectories from gym environment.
  """
  global global_step

  # Unpack inputs.
  env = input_data["env"]
  actor = input_data["actor"]
  critic = input_data["critic"]

  # Initialise variables.
  obsv, _ = env.reset()
  trajectory_data = {"states": [],
                     "actions": [],
                     "action_probabilities": [],
                     "rewards": [],
                     "values": [],
                     "terminals": [],
                     "actor_hidden_states": [],
                     #  "actor_cell_states": [],
                     "critic_hidden_states": [],
                     #  "critic_cell_states": [],
                     "prev_actions": [],
                     "prev_rewards": [],
                     #  "pulse_indices": [],
                     }
  terminal = torch.ones(hp.parallel_rollouts)

  with torch.no_grad():
    # Reset actor and critic state.
    actor.get_init_state(hp.parallel_rollouts, GATHER_DEVICE)
    critic.get_init_state(hp.parallel_rollouts, GATHER_DEVICE)
    # Take 1 additional step in order to collect the state and value for the final state.
    for i in range(hp.rollout_steps):

      trajectory_data["actor_hidden_states"].append(
          actor.hidden_cell.squeeze(0).cpu())
      trajectory_data["critic_hidden_states"].append(
          critic.hidden_cell.squeeze(0).cpu())

      # Get previous action
      if i == 0:
        prev_action = torch.zeros(hp.parallel_rollouts, actor.action_dim)
        prev_reward = torch.zeros(hp.parallel_rollouts, 1)
      else:
        prev_action = trajectory_data["actions"][-1]
        prev_reward = trajectory_data["rewards"][-1].unsqueeze(-1)
      # TODO: Add previous action to actor forward()
      # Choose next action
      state = torch.tensor(obsv, dtype=torch.float32)
      value = critic(
          state.unsqueeze(0).to(GATHER_DEVICE),
          prev_action.unsqueeze(0).to(GATHER_DEVICE),
          prev_reward.unsqueeze(0).to(GATHER_DEVICE),
          terminal.to(GATHER_DEVICE))
      action_dist = actor(
          state.unsqueeze(0).to(GATHER_DEVICE),
          prev_action.unsqueeze(0).to(GATHER_DEVICE),
          prev_reward.unsqueeze(0).to(GATHER_DEVICE),
          terminal.to(GATHER_DEVICE))
      action = action_dist.sample().reshape(hp.parallel_rollouts, -1)
      if not actor.continuous_action_space:
        action = action.squeeze(1)

      # Step environment
      action_np = action.cpu().numpy()
      obsv, reward, term, trunc, infos = env.step(action_np)
      done = np.logical_or(term, trunc)
      terminal = torch.tensor(done).float()

      trajectory_data["states"].append(state)
      trajectory_data["values"].append(value.squeeze(1).cpu())
      trajectory_data["actions"].append(action.cpu())
      trajectory_data["prev_actions"].append(prev_action.cpu())
      trajectory_data["prev_rewards"].append(prev_reward.cpu())
      trajectory_data["action_probabilities"].append(
          action_dist.log_prob(action).cpu())
      trajectory_data["rewards"].append(torch.tensor(reward).float())
      trajectory_data["terminals"].append(terminal)
      global_step += PARALLEL_ROLLOUTS

      if "final_info" not in infos:
        continue

      log_metrics(infos, writer, global_step)

    # Compute final value to allow for incomplete episodes.
    state = torch.tensor(obsv, dtype=torch.float32)
    value = critic(state=state.unsqueeze(0).to(GATHER_DEVICE),
                   prev_action=prev_action.unsqueeze(0).to(GATHER_DEVICE),
                   prev_reward=prev_reward.unsqueeze(0).to(GATHER_DEVICE),
                   terminal=terminal.to(GATHER_DEVICE))
    # Future value for terminal episodes is 0.
    trajectory_data["values"].append(value.squeeze(1).cpu() * (1 - terminal))

  # Combine step lists into tensors.
  trajectory_tensors = {key: torch.stack(
      value) for key, value in trajectory_data.items()}
  return trajectory_tensors

# %%


def split_trajectories_episodes(trajectory_tensors):
  """
  Split trajectories by episode.
  """

  len_episodes = []
  trajectory_episodes = {key: [] for key in trajectory_tensors.keys()}
  for i in range(hp.parallel_rollouts):
    terminals_tmp = trajectory_tensors["terminals"].clone()
    terminals_tmp[0, i] = 1
    terminals_tmp[-1, i] = 1
    split_points = (terminals_tmp[:, i] == 1).nonzero() + 1

    split_lens = split_points[1:] - split_points[:-1]
    split_lens[0] += 1

    len_episode = [split_len.item() for split_len in split_lens]
    len_episodes += len_episode
    for key, value in trajectory_tensors.items():
      # Value includes additional step.
      if key == "values":
        value_split = list(torch.split(
            value[:, i], len_episode[:-1] + [len_episode[-1] + 1]))
        # Append extra 0 to values to represent no future reward.
        for j in range(len(value_split) - 1):
          value_split[j] = torch.cat((value_split[j], torch.zeros(1)))
        trajectory_episodes[key] += value_split
      else:
        trajectory_episodes[key] += torch.split(value[:, i], len_episode)
  return trajectory_episodes, len_episodes

# %%


def pad_and_compute_returns(trajectory_episodes, len_episodes):
  """
  Pad the trajectories up to hp.rollout_steps so they can be combined in a
  single tensor.
  Add advantages and discounted_returns to trajectories.
  """

  episode_count = len(len_episodes)
  advantages_episodes, discounted_returns_episodes = [], []
  padded_trajectories = {key: [] for key in trajectory_episodes.keys()}
  padded_trajectories["advantages"] = []
  padded_trajectories["discounted_returns"] = []

  for i in range(episode_count):
    single_padding = torch.zeros(hp.rollout_steps - len_episodes[i])
    for key, value in trajectory_episodes.items():
      if value[i].ndim > 1:
        # padding = torch.zeros(hp.rollout_steps - len_episodes[i], value[0].shape[1], dtype=value[i].dtype)
        padding = torch.zeros(
            hp.rollout_steps - len_episodes[i], *value[i].shape[1:], dtype=value[i].dtype)
      else:
        padding = torch.zeros(
            hp.rollout_steps - len_episodes[i], dtype=value[i].dtype)
      padded_trajectories[key].append(torch.cat((value[i], padding)))
    padded_trajectories["advantages"].append(torch.cat((compute_advantages(rewards=trajectory_episodes["rewards"][i],
                                                                           values=trajectory_episodes["values"][i],
                                                                           discount=DISCOUNT,
                                                                           gae_lambda=GAE_LAMBDA), single_padding)))
    padded_trajectories["discounted_returns"].append(torch.cat((calc_discounted_return(rewards=trajectory_episodes["rewards"][i],
                                                                discount=DISCOUNT,
                                                                final_value=trajectory_episodes["values"][i][-1]), single_padding)))
  return_val = {k: torch.stack(v) for k, v in padded_trajectories.items()}
  return_val["seq_len"] = torch.tensor(len_episodes)

  return return_val

# %% [markdown]
# # Training dataset from trajectories

# %%


@dataclass
class TrajectoryBatch():
  """
  Dataclass for storing data batch.
  """
  states: torch.tensor
  actions: torch.tensor
  action_probabilities: torch.tensor
  advantages: torch.tensor
  discounted_returns: torch.tensor
  batch_size: torch.tensor
  actor_hidden_states: torch.tensor
  critic_hidden_states: torch.tensor
  prev_actions: torch.tensor
  prev_rewards: torch.tensor

# %%


class TrajectoryDataset():
  """
  Fast dataset for producing training batches from trajectories.
  """

  def __init__(self, trajectories, batch_size, device, batch_len):

    # Combine multiple trajectories into
    self.trajectories = {key: value.to(device)
                         for key, value in trajectories.items()}
    self.batch_len = batch_len
    truncated_seq_len = torch.clamp(
        trajectories["seq_len"] - batch_len + 1, 0, hp.rollout_steps)
    self.cumsum_seq_len = np.cumsum(np.concatenate(
        (np.array([0]), truncated_seq_len.numpy())))
    self.batch_size = batch_size

  def __iter__(self):
    self.valid_idx = np.arange(self.cumsum_seq_len[-1])
    self.batch_count = 0
    return self

  def __next__(self):
    if self.batch_count * self.batch_size >= math.ceil(self.cumsum_seq_len[-1] / self.batch_len):
      raise StopIteration
    else:
      actual_batch_size = min(len(self.valid_idx), self.batch_size)
      start_idx = np.random.choice(
          self.valid_idx, size=actual_batch_size, replace=False)
      self.valid_idx = np.setdiff1d(self.valid_idx, start_idx)
      eps_idx = np.digitize(
          start_idx, bins=self.cumsum_seq_len, right=False) - 1
      seq_idx = start_idx - self.cumsum_seq_len[eps_idx]
      series_idx = np.linspace(
          seq_idx, seq_idx + self.batch_len - 1, num=self.batch_len, dtype=np.int64)
      self.batch_count += 1
      return TrajectoryBatch(**{key: value[eps_idx, series_idx]for key, value
                                in self.trajectories.items() if key in TrajectoryBatch.__dataclass_fields__.keys()},
                             batch_size=actual_batch_size)

# %% [markdown]
# # PPO Training


def make_env(env_id, idx, gamma, seed):
  def thunk():
    env = gym.make(env_id, seed=seed+idx, **ENV_KWARGS)
    # env = TakeFirstN(env, TAKE_FIRST_N)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env = gym.wrappers.FlattenObservation(env)
    # env = AddPulseIndex(env)
    env = SquashAction(
        env) if adapt_weight < 1 else gym.wrappers.ClipAction(env)
    # env = gym.wrappers.NormalizeReward(env, gamma=gamma)
    # env = gym.wrappers.TransformReward(
    #     env, lambda reward: np.clip(reward, -10, 10))
    return env

  return thunk

# %%


def train_model(actor, critic, actor_optimizer, critic_optimizer, iteration, stop_conditions):

    # Vector environment manages multiple instances of the environment.
    # A key difference between this and the standard gym environment is it automatically resets.
    # Therefore when the done flag is active in the done vector the corresponding state is the first new state.
  env = gym.vector.AsyncVectorEnv(
      [make_env(ENV, i, DISCOUNT, RANDOM_SEED) for i in range(hp.parallel_rollouts)])
  while iteration < stop_conditions.max_iterations:

    actor = actor.to(GATHER_DEVICE)
    critic = critic.to(GATHER_DEVICE)
    start_gather_time = time.time()

    # Gather trajectories.
    input_data = {"env": env, "actor": actor, "critic": critic, "discount": hp.discount,
                  "gae_lambda": hp.gae_lambda}
    trajectory_tensors = gather_trajectories(input_data)
    trajectory_episodes, len_episodes = split_trajectories_episodes(
        trajectory_tensors)
    trajectories = pad_and_compute_returns(trajectory_episodes, len_episodes)

    # Calculate mean reward.
    complete_episode_count = trajectories["terminals"].sum().item()
    terminal_episodes_rewards = (trajectories["terminals"].sum(
        axis=1) * trajectories["rewards"].sum(axis=1)).sum()
    mean_reward = terminal_episodes_rewards / (complete_episode_count)

    # Check stop conditions.
    if mean_reward > stop_conditions.best_reward:
      stop_conditions.best_reward = mean_reward
      stop_conditions.fail_to_improve_count = 0
    else:
      stop_conditions.fail_to_improve_count += 1
    if stop_conditions.fail_to_improve_count > hp.patience:
      print(
          f"Policy has not yielded higher reward for {hp.patience} iterations...  Stopping now.")
      break

    trajectory_dataset = TrajectoryDataset(trajectories, batch_size=hp.batch_size,
                                           device=TRAIN_DEVICE, batch_len=hp.recurrent_seq_len)
    end_gather_time = time.time()
    start_train_time = time.time()

    actor = actor.to(TRAIN_DEVICE)
    critic = critic.to(TRAIN_DEVICE)

    # Train actor and critic.
    for epoch_idx in range(hp.ppo_epochs):
      for batch in trajectory_dataset:

        # Get batch
        actor.hidden_cell = batch.actor_hidden_states[:1]

        # Update actor
        actor_optimizer.zero_grad()
        action_dist = actor(state=batch.states, 
                            prev_action=batch.prev_actions,
                            prev_reward=batch.prev_rewards,
                            )
        # Action dist runs on cpu as a workaround to CUDA illegal memory access.
        action_probabilities = action_dist.log_prob(
            batch.actions[-1, :].to("cpu")).to(TRAIN_DEVICE)
        # Compute probability ratio from probabilities in logspace.
        probabilities_ratio = torch.exp(
            action_probabilities - batch.action_probabilities[-1, :])
        surrogate_loss_0 = probabilities_ratio * batch.advantages[-1, :]
        surrogate_loss_1 = torch.clamp(
            probabilities_ratio, 1. - hp.ppo_clip, 1. + hp.ppo_clip) * batch.advantages[-1, :]
        surrogate_loss_2 = action_dist.entropy().to(TRAIN_DEVICE)
        actor_loss = -torch.mean(torch.min(surrogate_loss_0, surrogate_loss_1)) - \
            torch.mean(hp.entropy_factor * surrogate_loss_2)
        actor_loss.backward()
        torch.nn.utils.clip_grad.clip_grad_norm_(
            actor.parameters(), hp.max_grad_norm)
        actor_optimizer.step()

        # Update critic
        critic_optimizer.zero_grad()
        critic.hidden_cell = batch.critic_hidden_states[:1]
        values = critic(state=batch.states, 
                        prev_action=batch.prev_actions,
                        prev_reward=batch.prev_rewards,)
        critic_loss = F.mse_loss(
            batch.discounted_returns[-1, :], values.squeeze(1))
        torch.nn.utils.clip_grad.clip_grad_norm_(
            critic.parameters(), hp.max_grad_norm)
        critic_loss.backward()
        critic_optimizer.step()

    end_train_time = time.time()
    print(f"Iteration: {iteration},  Mean reward: {mean_reward}, Mean Entropy: {torch.mean(surrogate_loss_2)}, " +
          f"complete_episode_count: {complete_episode_count}, Gather time: {end_gather_time - start_gather_time:.2f}s, " +
          f"Train time: {end_train_time - start_train_time:.2f}s")

    if SAVE_METRICS_TENSORBOARD:
      writer.add_scalar("complete_episode_count",
                        complete_episode_count, iteration)
      writer.add_scalar("total_reward", mean_reward, iteration)
      writer.add_scalar("actor_loss", actor_loss, iteration)
      writer.add_scalar("critic_loss", critic_loss, iteration)
      writer.add_scalar("policy_entropy", torch.mean(
          surrogate_loss_2), iteration)
    if SAVE_PARAMETERS_TENSORBOARD:
      save_parameters(writer, "actor", actor, iteration)
      save_parameters(writer, "value", critic, iteration)
    if iteration % CHECKPOINT_FREQUENCY == 0:
      save_checkpoint(actor, critic, actor_optimizer,
                      critic_optimizer, iteration, stop_conditions)
    iteration += 1

  return stop_conditions.best_reward

# %% [markdown]
# # Environment Setup

# %% [markdown]
# # Run training


# %%
# from mpar_sim.envs import SpectrumEnv
# gym.envs.register(id='SpectrumEnv', entry_point=SpectrumEnv)
if __name__ == '__main__':
  import mpar_sim.envs
  writer = SummaryWriter(
      log_dir=f"{WORKSPACE_PATH}/logs/ppo_continuous/{EXPERIMENT_NAME}")
  actor, critic, actor_optimizer, critic_optimizer, iteration, stop_conditions = start_or_resume_from_checkpoint()
  score = train_model(actor, critic, actor_optimizer,
                      critic_optimizer, iteration, stop_conditions)

# %%
