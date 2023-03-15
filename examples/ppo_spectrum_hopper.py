# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_ataripy
from mpar_sim.models.networks.impala_cnn import ImpalaNetwork
import mpar_sim.envs
from torch.utils.tensorboard import SummaryWriter
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical
from lightning_rl.models.on_policy.ppo import ppo_loss
from lightning_rl.common.advantage import estimate_advantage
import torch.optim as optim
import torch.nn as nn
import torch
import numpy as np
import gymnasium as gym
from distutils.util import strtobool
from collections import namedtuple
import time
import random
import argparse
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# import gymnasium as gym


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
  parser.add_argument("--wandb-project-name", type=str, default="cleanRL",
      help="the wandb's project name")
  parser.add_argument("--wandb-entity", type=str, default=None,
      help="the entity (team) of wandb's project")
  parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
      help="whether to capture videos of the agent performances (check out `videos` folder)")

  # Algorithm specific arguments
  parser.add_argument("--env-id", type=str, default="mpar_sim/SpectrumHopper-v0",
      help="the id of the environment")
  parser.add_argument("--total-timesteps", type=int, default=1_000_000,
      help="total timesteps of the experiments")
  parser.add_argument("--learning-rate", type=float, default=3e-4,
      help="the learning rate of the optimizer")
  parser.add_argument("--num-envs", type=int, default=16,
      help="the number of parallel game environments")
  parser.add_argument("--num-steps", type=int, default=2048,
      help="the number of steps to run in each environment per policy rollout")
  parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
      help="Toggle learning rate annealing for policy and value networks")
  parser.add_argument("--gamma", type=float, default=0.99,
      help="the discount factor gamma")
  parser.add_argument("--gae-lambda", type=float, default=0.95,
      help="the lambda for the general advantage estimation")
  parser.add_argument("--num-minibatches", type=int, default=4,
      help="the number of mini-batches")
  parser.add_argument("--update-epochs", type=int, default=10,
      help="the K epochs to update the policy")
  parser.add_argument("--norm-adv", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
      help="Toggles advantages normalization")
  parser.add_argument("--clip-coef", type=float, default=0.2,
      help="the surrogate clipping coefficient")
  parser.add_argument("--clip-vloss", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
      help="Toggles whether or not to use a clipped loss for the value function, as per the paper.")
  parser.add_argument("--ent-coef", type=float, default=0.0,
      help="coefficient of the entropy")
  parser.add_argument("--vf-coef", type=float, default=1,
      help="coefficient of the value function")
  parser.add_argument("--max-grad-norm", type=float, default=0.5,
      help="the maximum norm for the gradient clipping")
  parser.add_argument("--target-kl", type=float, default=None,
      help="the target KL divergence threshold")
  args = parser.parse_args()
  args.batch_size = int(args.num_envs * args.num_steps)
  args.minibatch_size = int(args.batch_size // args.num_minibatches)
  # fmt: on
  return args


def make_env(env_id, seed, idx, capture_video, run_name, gamma):
  """TODO: Customize this."""
  def thunk():
    if capture_video:
      env = gym.make(env_id, render_mode="rgb_array")
    else:
      env = gym.make(env_id)
    if capture_video:
      if idx == 0:
        env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
    env = gym.wrappers.TimeLimit(env, max_episode_steps=250)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env = gym.wrappers.ClipAction(env)
    env = gym.wrappers.NormalizeReward(env, gamma=gamma)
    env = gym.wrappers.TransformReward(
        env, lambda reward: np.clip(reward, -10, 10))
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    return env
  return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
  torch.nn.init.orthogonal_(layer.weight, std)
  torch.nn.init.constant_(layer.bias, bias_const)
  return layer


class Agent(nn.Module):
  def __init__(self, envs, hidden_size: int = 256):
    super().__init__()
    self.impala_net = ImpalaNetwork(*envs.single_observation_space.shape,
                                    out_features=hidden_size)

    n_actions = np.prod(envs.single_action_space.shape)
    self.actor_mean = layer_init(nn.Linear(hidden_size, n_actions), std=0.01)
    self.actor_logstd = nn.Parameter(torch.zeros(1, n_actions))
    self.critic = layer_init(nn.Linear(hidden_size, 1), std=1.0)

  def get_value(self, x: torch.Tensor):
    features = self.impala_net(x / 255.0)
    return self.critic(features)

  def get_action_and_value(self, x, action=None):
    features = self.impala_net(x / 255.0)
    action_mean = self.actor_mean(features)
    action_logstd = self.actor_logstd.expand_as(action_mean)
    action_std = torch.exp(action_logstd)
    probs = Normal(action_mean, action_std)
    if action is None:
      action = probs.sample()
    return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(features)


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
  envs = gym.vector.AsyncVectorEnv(
      [make_env(args.env_id, args.seed + i, i, args.capture_video, run_name, args.gamma)
       for i in range(args.num_envs)]
  )
  # envs = gym.wrappers.RecordEpisodeStatistics(envs)
  assert isinstance(envs.single_action_space,
                    gym.spaces.Box), "only continuous action space is supported"

  agent = Agent(envs).to(device)
  optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

  # ALGO Logic: Storage setup
  obs = torch.zeros((args.num_steps, args.num_envs) +
                    envs.single_observation_space.shape).to(device)
  actions = torch.zeros((args.num_steps, args.num_envs) +
                        envs.single_action_space.shape).to(device)
  logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
  rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
  dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
  values = torch.zeros((args.num_steps, args.num_envs)).to(device)

  # TRY NOT TO MODIFY: start the game
  global_step = 0
  start_time = time.time()
  next_obs = torch.Tensor(envs.reset()[0]).to(device)
  next_done = torch.zeros(args.num_envs).to(device)
  num_updates = args.total_timesteps // args.batch_size

  for update in range(1, num_updates + 1):
    # Annealing the rate if instructed to do so.
    if args.anneal_lr:
      lr_frac = 1.0 - (update - 1.0) / num_updates
      current_lr = lr_frac * args.learning_rate
      optimizer.param_groups[0]["lr"] = current_lr

    for step in range(0, args.num_steps):
      global_step += 1 * args.num_envs
      obs[step] = next_obs
      dones[step] = next_done

      # ALGO LOGIC: action logic
      with torch.no_grad():
        action, logprob, _, value = agent.get_action_and_value(next_obs)
        values[step] = value.flatten()
      actions[step] = action
      logprobs[step] = logprob

      # TRY NOT TO MODIFY: execute the game and log data.
      next_obs, reward, terminated, truncated, infos = envs.step(
          action.cpu().numpy())
      done = np.logical_or(terminated, truncated)
      rewards[step] = torch.tensor(reward).to(device).view(-1)
      next_obs, next_done = torch.Tensor(next_obs).to(
          device), torch.Tensor(done).to(device)
      next_value = agent.get_value(next_obs).reshape(1, -1)

      # Only print when at least 1 env is done
      if "final_info" not in infos:
        continue

      mean_episodic_return = np.mean(
          [info["episode"]["r"] for info in infos["final_info"] if info is not None])
      mean_episodic_length = np.mean(
          [info["episode"]["l"] for info in infos["final_info"] if info is not None])
      print(
          f"global_step={global_step}, mean_return={mean_episodic_return:.2f}, mean_length={mean_episodic_length:.2f}")
      writer.add_scalar("charts/episodic_return",
                        mean_episodic_return, global_step)
      writer.add_scalar("charts/episodic_length",
                        mean_episodic_length, global_step)

    # bootstrap value if not done
    advantages, returns = estimate_advantage(
        rewards=rewards,
        values=values,
        dones=dones,
        last_value=next_value,
        last_done=next_done,
        n_steps=args.num_steps,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        normalize_return=False,
        normalize_advantage=False,
    )

    # flatten the batch
    batch_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
    batch_log_probs = logprobs.reshape(-1)
    batch_actions = actions.reshape((-1,) + envs.single_action_space.shape)
    batch_advantages = advantages.reshape(-1)
    batch_returns = returns.reshape(-1)
    batch_values = values.reshape(-1)

    # Optimizing the policy and value network
    batch_inds = np.arange(args.batch_size)
    for epoch in range(args.update_epochs):
      np.random.shuffle(batch_inds)
      for start in range(0, args.batch_size, args.minibatch_size):
        end = start + args.minibatch_size
        minibatch_inds = batch_inds[start:end]

        _, new_log_probs, entropy, new_values = agent.get_action_and_value(
            batch_obs[minibatch_inds], batch_actions.long()[minibatch_inds])
        loss, info = ppo_loss(
            batch_advantages=batch_advantages[minibatch_inds],
            batch_log_probs=batch_log_probs[minibatch_inds],
            batch_values=batch_values[minibatch_inds],
            batch_returns=batch_returns[minibatch_inds],
            new_log_probs=new_log_probs,
            new_values=new_values.flatten(),
            entropy=entropy,
            clip_range=args.clip_coef,
            value_coef=args.vf_coef,
            entropy_coef=args.ent_coef,
            normalize_advantage=args.norm_adv,
        )

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
        optimizer.step()

      if args.target_kl is not None and info['metrics/approx_kl'] > args.target_kl:
        break

    # TRY NOT TO MODIFY: record rewards for plotting purposes
    for key, value in info.items():
      writer.add_scalar(key, value, global_step)

  envs.close()
  writer.close()
