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
from mpar_sim.particle.surveillance_pso import SurveillanceSwarm
from mpar_sim.radar import PhasedArrayRadar
from mpar_sim.wrappers.squeeze_image import SqueezeImage
from lightning_rl.common.layer_init import ortho_init


seed = np.random.randint(0, 2**32 - 1)
warnings.filterwarnings("ignore")
plt.rcParams['font.size'] = 14
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"

#############################
# Environment definition
#############################


def make_env(env_id,
             radar,
             max_episode_steps):
  def thunk():
    # In this experiment, targets move according to a constant velocity, white noise acceleration model.
    # http://www.control.isy.liu.se/student/graduate/targettracking/file/le2_handout.pdf
    transition_model = ConstantVelocity(ndim_pos=3, noise_diff_coeff=10)

    pos_bounds = np.array([[radar.az_fov[0], radar.el_fov[0]],
                           [radar.az_fov[1], radar.el_fov[1]]])
    swarm = SurveillanceSwarm(n_particles=5_000,
                              n_dimensions=2,
                              position_bounds=pos_bounds,
                              velocity_bounds=[-1, 1],
                              gravity=0.075,
                              min_dispersion_inertia=0.25,
                              max_dispersion_inertia=0.95,
                              detection_inertia=0.25,
                              )

    env = gym.make(env_id,
                   radar=radar,
                   swarm=swarm,
                   transition_model=transition_model,
                   min_initial_n_targets=50,
                   max_initial_n_targets=50,
                   max_az_span=40,
                   max_el_span=40,
                   range_span=[10e3, 30e3],
                   velocity_span=[-100, 100],
                   birth_rate=0,
                   death_probability=0,
                   randomize_initial_state=True,
                   n_confirm_detections=3,
                   mutation_rate=0,
                   render_mode='rgb_array',
                   n_obs_bins=100,
                   )

    # Wrappers
    env = gym.wrappers.FlattenObservation(env)
    env = gym.wrappers.FrameStack(env, 4)
    env = gym.wrappers.TimeLimit(env, max_episode_steps=max_episode_steps)
    env = gym.wrappers.ClipAction(env)

    return env

  return thunk


# %% [markdown]
# ## Environment setup
# %%
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
# Create the environment
env_id = 'mpar_sim/SimpleParticleSurveillance-v0'
n_env = 16
max_episode_steps = 1500
env = gym.vector.AsyncVectorEnv(
    [make_env(env_id,  radar, max_episode_steps) for _ in range(n_env)])
env = gym.wrappers.RecordEpisodeStatistics(env=env, deque_size=20)

#############################
# Agent definitions
#############################
az_bw = 5
el_bw = 5
bw = 100e6
pulsewidth = 10e-6
prf = 5e3
n_pulses = 32

raster_agent = RasterScanAgent(
    azimuth_scan_limits=np.array([-45, 45]),
    elevation_scan_limits=np.array([-45, 45]),
    azimuth_beam_spacing=0.75,
    elevation_beam_spacing=0.75,
    azimuth_beamwidth=az_bw,
    elevation_beamwidth=el_bw,
    bandwidth=bw,
    pulsewidth=pulsewidth,
    prf=prf,
    n_pulses=n_pulses,
)


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
        ortho_init(nn.Conv2d(self.observation_space.shape[0], 1, 1)),
        nn.Flatten(start_dim=1, end_dim=-1),
        nn.Tanh(),
        ortho_init(nn.Linear(self.observation_space.shape[1], 64)),
        nn.Tanh(),
        ortho_init(nn.Linear(64, 2 * len(self.stochastic_action_inds)), std=0.01)
    )
    self.critic = nn.Sequential(
        ortho_init(nn.Conv2d(self.observation_space.shape[0], 1, 1)),
        nn.Flatten(start_dim=1, end_dim=-1),
        nn.Tanh(),
        ortho_init(nn.Linear(self.observation_space.shape[1], 64)),
        nn.Tanh(),
        ortho_init(nn.Linear(64, 1), std=1.0),
    )

    self.save_hyperparameters()

  def forward(self, x: torch.Tensor):
    # Sample the action from its distribution
    actor_out = self.actor(x.view(x.shape + (1,)))
    actor_out = list(torch.chunk(actor_out, 2, dim=1))
    mean, var = actor_out[0], actor_out[1]
    cov = torch.diag_embed(torch.exp(0.5*torch.clamp(var, -5, 5)))
    # Compute the value of the state
    value = self.critic(x.view(x.shape + (1,))).flatten()

    return mean, cov, value

  def act(self, observations: torch.Tensor):
    mean, cov, value = self.forward(observations)
    action_dist = torch.distributions.MultivariateNormal(mean, cov)
    stochastic_actions = action_dist.sample()
    # TODO: Use the stochastic action indices to determine the action order
    n_envs = observations.shape[0]
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


checkpoint_filename = "/home/shane/src/mpar-sim/lightning_logs/version_540/checkpoints/epoch=84-step=10200.ckpt"
ppo_agent = PPOSurveillanceAgent.load_from_checkpoint(
    checkpoint_filename, env=env, seed=seed)


#############################
# Experiement setup
#############################
# Metrics
ppo_init_ratio = np.ones((max_episode_steps, n_env))
raster_init_ratio = np.ones((max_episode_steps, n_env))
random_init_ratio = np.ones((max_episode_steps, n_env))
spso_init_ratio = np.ones((max_episode_steps, n_env))

beam_coverage_map = np.zeros((32, 32))
az_axis = np.linspace(-45, 45, beam_coverage_map.shape[1])
el_axis = np.linspace(-45, 45, beam_coverage_map.shape[0])
az_pixel_width = az_axis[1] - az_axis[0]
el_pixel_width = el_axis[1] - el_axis[0]

tic = time.time()
obs, info = env.reset(seed=seed)
dones = np.zeros(n_env, dtype=bool)
i = 0
with torch.no_grad():
  while not np.all(dones):
    obs_tensor = torch.as_tensor(obs).to(
        device=ppo_agent.device, dtype=torch.float32)
    action_tensor = ppo_agent.act(obs_tensor)[0]
    actions = action_tensor.detach().cpu().numpy()
    # Repeat actions for all environments
    obs, reward, terminated, truncated, info = env.step(actions)
    dones = np.logical_or(dones, np.logical_or(terminated, truncated))

    ppo_init_ratio[i, ~dones] = info['initiation_ratio'][~dones]

    i += 1
toc = time.time()
print("PPO agent done")
print(f"Time elapsed: {toc-tic:.2f} seconds")

# Deterministic swarm agent
tic = time.time()
obs, info = env.reset(seed=seed)
dones = np.zeros(n_env, dtype=bool)
i = 0
while not np.all(dones):
  # Perform an environment step
  look = raster_agent.act(obs)
  az_beamwidth = (look.azimuth_beamwidth - radar.min_az_beamwidth) / \
      (radar.max_az_beamwidth - radar.min_az_beamwidth)
  el_beamwidth = (look.elevation_beamwidth - radar.min_el_beamwidth) / \
      (radar.max_el_beamwidth - radar.min_el_beamwidth)
  actions = np.array([[0, 0,
                     az_beamwidth,
                     el_beamwidth,
                       look.bandwidth,
                       look.pulsewidth,
                       look.prf,
                       look.n_pulses]])
  actions = np.repeat(actions, n_env, axis=0)
  actions[:, 0] = obs[:, 0, 0]
  actions[:, 1] = obs[:, 0, 1]
  obs, reward, terminated, truncated, info = env.step(actions)
  dones = np.logical_or(dones, np.logical_or(terminated, truncated))

  # Update metrics
  spso_init_ratio[i, ~dones] = info['initiation_ratio'][~dones]
  if not dones[0]:
    n_az_pixels = int(np.ceil(look.azimuth_beamwidth / az_pixel_width) // 2)
    n_el_pixels = int(np.ceil(look.elevation_beamwidth / el_pixel_width) // 2)
    actions[0, :2] = actions[0, :2]*45
    az = np.digitize(actions[0, 0], az_axis, right=True)
    el = np.digitize(actions[0, 1], el_axis[::-1], right=True)
    beam_coverage_map[max(el-n_el_pixels, 0):min(el+n_el_pixels, len(el_axis)),
                      max(az-n_az_pixels, 0):min(az+n_az_pixels, len(az_axis))] += 1

  i += 1
toc = time.time()
print("Deterministic swarm agent done")
print(f"Time elapsed: {toc-tic:.2f} seconds")

# Raster agent
tic = time.time()
obs, info = env.reset(seed=seed)
dones = np.zeros(n_env, dtype=bool)
i = 0
while not np.all(dones):
  # Perform an environment step
  look = raster_agent.act(obs)
  az_beamwidth = (look.azimuth_beamwidth - radar.min_az_beamwidth) / \
      (radar.max_az_beamwidth - radar.min_az_beamwidth)
  el_beamwidth = (look.elevation_beamwidth - radar.min_el_beamwidth) / \
      (radar.max_el_beamwidth - radar.min_el_beamwidth)
  actions = np.array([[look.azimuth_steering_angle / 45,
                     look.elevation_steering_angle / 45,
                     az_beamwidth,
                     el_beamwidth,
                     look.bandwidth,
                     look.pulsewidth,
                     look.prf,
                     look.n_pulses]])
  actions = np.repeat(actions, n_env, axis=0)
  obs, reward, terminated, truncated, info = env.step(actions)
  dones = np.logical_or(dones, np.logical_or(terminated, truncated))

  # Update metrics
  raster_init_ratio[i, ~dones] = info['initiation_ratio'][~dones]
  i += 1
toc = time.time()
print("Raster agent done")
print(f"Time elapsed: {toc-tic:.2f} seconds")

# Uniform random agent
tic = time.time()
obs, info = env.reset(seed=seed)
dones = np.zeros(n_env, dtype=bool)
i = 0
while not np.all(dones):
  # Perform an environment step
  look = raster_agent.act(obs)
  az_beamwidth = (look.azimuth_beamwidth - radar.min_az_beamwidth) / \
      (radar.max_az_beamwidth - radar.min_az_beamwidth)
  el_beamwidth = (look.elevation_beamwidth - radar.min_el_beamwidth) / \
      (radar.max_el_beamwidth - radar.min_el_beamwidth)
  actions = np.array([[np.random.uniform(-1, 1),
                     np.random.uniform(-1, 1),
                     az_beamwidth,
                     el_beamwidth,
                     look.bandwidth,
                     look.pulsewidth,
                     look.prf,
                     look.n_pulses]])
  actions = np.repeat(actions, n_env, axis=0)
  obs, reward, terminated, truncated, info = env.step(actions)
  dones = np.logical_or(dones, np.logical_or(terminated, truncated))

  # Update metrics
  random_init_ratio[i, ~dones] = info['initiation_ratio'][~dones]
  i += 1
toc = time.time()
print("Random agent done")
print(f"Time elapsed: {toc-tic:.2f} seconds")

#############################
# Plot results
#############################
fig, ax = plt.subplots()
plt.plot(np.mean(raster_init_ratio[:-1, :], axis=1), linewidth=2)
plt.plot(np.mean(random_init_ratio[:-1, :], axis=1), linewidth=2)
plt.plot(np.mean(spso_init_ratio[:-1, :], axis=1), linewidth=2)
plt.plot(np.mean(ppo_init_ratio[:-1, :], axis=1), linewidth=2)
plt.grid()
plt.xlabel('Time step (dwells)', fontsize=14)
plt.ylabel('Fraction of targets under track', fontsize=14)
plt.legend(['Raster', 'Random', 'SPSO', 'RL'], fontsize=14)

plt.figure()
plt.imshow(beam_coverage_map,
           norm='linear',
           extent=[az_axis[0], az_axis[-1], el_axis[0], el_axis[-1]])
plt.xlabel('Azimuth (degrees)')
plt.ylabel('Elevation (degrees)')
plt.colorbar()

plt.show()
env.close()

# %%
