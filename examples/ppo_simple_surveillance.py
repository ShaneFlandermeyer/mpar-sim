# %% [markdown]
# ## Simple Particle Surveillance Environment

# %% [markdown]
# ## Imports

# %%
import time
from typing import Tuple
import warnings

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
from lightning_rl.models.on_policy_models.ppo import PPO
from mpar_sim.models.transition.linear import ConstantVelocity
from stonesoup.types.state import GaussianState
from torch import nn

import mpar_sim.envs
from mpar_sim.agents.raster_scan import RasterScanAgent
from mpar_sim.beam.beam import SincBeam
from mpar_sim.common.wrap_to_interval import wrap_to_interval
from mpar_sim.radar import PhasedArrayRadar
from mpar_sim.wrappers.squeeze_image import SqueezeImage
from lightning_rl.common.layer_init import ortho_init


# %% [markdown]
# ## Setup

# %%
seed = np.random.randint(0, 2**32 - 1)
warnings.filterwarnings("ignore")
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"

# %% [markdown]
# ## Environment creation

# %%


def make_env(env_id,
             max_episode_steps):
  def thunk():
    # In this experiment, targets move according to a constant velocity, white noise acceleration model.
    # http://www.control.isy.liu.se/student/graduate/targettracking/file/le2_handout.pdf
    transition_model = ConstantVelocity(ndim_pos=3, noise_diff_coeff=10)
    # Radar system object
    radar = PhasedArrayRadar(
        ndim_state=6,
        position_mapping=[0, 2, 4],
        velocity_mapping=[1, 3, 5],
        position=np.array([[0], [0], [0]]),
        rotation_offset=np.array([[0], [0], [0]]),
        # Array parameters
        n_elements_x=32,
        n_elements_y=32,
        element_spacing=0.5,  # Wavelengths
        element_tx_power=10,
        max_az_beamwidth=10,
        max_el_beamwidth=10,
        # System parameters
        center_frequency=3e9,
        system_temperature=290,
        noise_figure=4,
        # Scan settings
        beam_shape=SincBeam,
        az_fov=[-45, 45],
        el_fov=[-45, 45],
        # Detection settings
        false_alarm_rate=1e-6,
        include_false_alarms=False
    )
    # Gaussian parameters used to initialize the states of new targets in the scene. Here, elements (0, 2, 4) of the state vector/covariance are the az/el/range of the target (angles in degrees), and (1, 3, 5) are the x/y/z velocities in m/s. If randomize_initial_state is set to True in the environment, the mean az/el are uniformly sampled across the radar field of view, and the variance is uniformly sampled from [0, max_random_az_covar] and [0, max_random_el_covar] for the az/el, respectively
    initial_state = GaussianState(
        state_vector=[30,   0,  -20,   0, 10e3, 0],
        covar=np.diag([3**2, 100**2, 3**2, 100**2,  1000**2, 100**2])
    )

    env = gym.make(env_id,
                   radar=radar,
                   transition_model=transition_model,
                   initial_state=initial_state,
                   birth_rate=0,
                   death_probability=0,
                   min_initial_n_targets=50,
                   max_initial_n_targets=50,
                   n_confirm_detections=3,
                   randomize_initial_state=True,
                   max_random_az_covar=10**2,
                   max_random_el_covar=10**2,
                   beta_g=0.075,
                   w_disp_min=0.25,
                   w_disp_max=0.95,
                   w_det=0.25,
                   mutation_rate=0,
                   render_mode='rgb_array',
                   n_obs_bins=50,
                   )

    # Wrappers
    env = gym.wrappers.FrameStack(env, 2)
    env = gym.wrappers.FlattenObservation(env)
    env = gym.wrappers.TimeLimit(env, max_episode_steps=max_episode_steps)
    env = gym.wrappers.ClipAction(env)

    return env

  return thunk


# %% [markdown]
# ## RL agent defintion

# %%


class PPOSurveillanceAgent(PPO):
  def __init__(self,
               env: gym.Env,
               # Look parameters 
               bandwidth: float,
               pulsewidth: float,
               prf: float,
               n_pulses: int,
               rpo_alpha: float = 0,
               **kwargs):
    super().__init__(env=env, **kwargs)
    # Populate look parameters
    self.bandwidth = bandwidth
    self.pulsewidth = pulsewidth
    self.prf = prf
    self.n_pulses = n_pulses

    self.rpo_alpha = rpo_alpha
    self.stochastic_action_inds = [0, 1, 2, 3]
    self.actor = nn.Sequential(
        ortho_init(nn.Linear(np.array(self.observation_space.shape).prod(), 64)),
        nn.Tanh(),
        ortho_init(nn.Linear(64, 64)),
        nn.Tanh(),
        ortho_init(nn.Linear(64, 2 * len(self.stochastic_action_inds)), std=0.01)
    )
    self.critic = nn.Sequential(
        ortho_init(
            nn.Linear(np.array(self.observation_space.shape).prod(), 64)),
        nn.Tanh(),
        ortho_init(nn.Linear(64, 64)),
        nn.Tanh(),
        ortho_init(nn.Linear(64, 1), std=1.0),
    )

    self.save_hyperparameters()

  def forward(self, x: torch.Tensor):
    # Sample the action from its distribution
    actor_out = self.actor(x)
    actor_out = list(torch.chunk(actor_out, 2, dim=1))
    mean, var = actor_out[0], actor_out[1]
    cov = torch.diag_embed(torch.exp(0.5*torch.clamp(var, -5, 5)))
    # Compute the value of the state
    value = self.critic(x).flatten()

    return mean, cov, value

  def act(self, observations: torch.Tensor):
    mean, cov, value = self.forward(observations)
    action_dist = torch.distributions.MultivariateNormal(mean, cov)
    stochastic_actions = action_dist.sample()
    # TODO: Use the stochastic action indices to determine the action order
    n_envs = observations.shape[0] if observations.ndim == 2 else 1
    deterministic_actions = (
        # Bandwidth
        torch.full((n_envs, 1), self.bandwidth).to(
            stochastic_actions.device),
        # Pulsewidth
        torch.full((n_envs, 1), self.pulsewidth).to(
            stochastic_actions.device),
        # PRF
        torch.full((n_envs, 1), self.prf).to(
            stochastic_actions.device),
        # Number of pulses
        torch.full((n_envs, 1), self.n_pulses).to(
            stochastic_actions.device),
    )
    action = torch.cat((stochastic_actions,) + deterministic_actions, 1)
    return action, value, action_dist.log_prob(stochastic_actions), action_dist.entropy()

  def evaluate_actions(self,
                       observations: torch.Tensor,
                       actions: torch.Tensor):
    # Only evaluate stochastic actions
    actions = actions[:, self.stochastic_action_inds]
    mean, cov, value = self.forward(observations)
    # Apply RPO to improve exploration
    if self.rpo_alpha > 0:
      z = torch.FloatTensor(
          mean.shape).uniform_(-self.rpo_alpha, self.rpo_alpha)
      mean = mean + z.to(mean.device)
    action_dist = torch.distributions.MultivariateNormal(mean, cov)
    return action_dist.log_prob(actions), action_dist.entropy(), value

  def configure_optimizers(self):
    optimizer = torch.optim.Adam(self.parameters(), lr=5e-4, eps=1e-5)
    return optimizer


# %% [markdown]
# ## Environment setup
# %%
# Create the environment
env_id = 'mpar_sim/SimpleParticleSurveillance-v0'
n_env = 16
max_episode_steps = 1000
env = gym.vector.AsyncVectorEnv(
    [make_env(env_id,  max_episode_steps) for _ in range(n_env)])
env = gym.wrappers.RecordEpisodeStatistics(env=env, deque_size=20)
# env = gym.wrappers.NormalizeReward(env)


# %% [markdown]
# ## Training loop

# %%
bw = 100e6
pulsewidth = 10e-6
prf = 5e3
n_pulses = 32

ppo_agent = PPOSurveillanceAgent(env,
                                 n_rollouts_per_epoch=3,
                                 n_steps_per_rollout=512,
                                 n_gradient_steps=10,
                                 batch_size=2048,
                                 gamma=0.99,
                                 gae_lambda=0.95,
                                 value_coef=0.5,
                                 entropy_coef=0,
                                 rpo_alpha=0,
                                 seed=seed,
                                 normalize_advantage=True,
                                 policy_clip_range=0.2,
                                 target_kl=None,
                                 # Radar parameters
                                 bandwidth=bw,
                                 pulsewidth=pulsewidth,
                                 prf=prf,
                                 n_pulses=n_pulses,
                                 )


# checkpoint_filename = "/home/shane/src/mpar-sim/lightning_logs/version_462/checkpoints/epoch=49-step=6000.ckpt"
# ppo_agent = PPOSurveillanceAgent.load_from_checkpoint(
#     checkpoint_filename, env=env, seed=seed)

trainer = pl.Trainer(
    max_epochs=50,
    gradient_clip_val=0.5,
    accelerator='gpu',
    devices=1,
)
trainer.fit(ppo_agent)
