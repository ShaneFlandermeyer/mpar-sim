# %% [markdown]
# ## Simple Particle Surveillance Environment

# %% [markdown]
# ## Imports

# %%
import copy
import time
from typing import Tuple
import warnings

import cv2
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
from lightning_rl.models.on_policy_models import PPO
from pyswarms.utils.functions import single_obj as fx
from pyswarms.utils.plotters import (plot_contour, plot_cost_history,
                                     plot_surface)
from pyswarms.utils.plotters.formatters import Designer, Mesher
from stonesoup.models.transition.linear import (
    CombinedLinearGaussianTransitionModel, ConstantVelocity, SingerApproximate)
from stonesoup.plotter import Dimension, Plotter
from stonesoup.types.array import CovarianceMatrix, StateVector
from stonesoup.types.state import GaussianState
from torch import distributions, nn

import mpar_sim.envs
from mpar_sim.agents.raster_scan import RasterScanAgent
from mpar_sim.beam.beam import GaussianBeam, RectangularBeam, SincBeam
from mpar_sim.common.wrap_to_interval import wrap_to_interval
from mpar_sim.defaults import (default_gbest_pso, default_lbest_pso,
                               default_radar, default_raster_scan_agent,
                               default_scheduler)
from mpar_sim.models.motion.constant_velocity import constant_velocity
from mpar_sim.radar import PhasedArrayRadar
from mpar_sim.wrappers.image_to_pytorch import ImageToPytorch
from mpar_sim.wrappers.squeeze_image import SqueezeImage


# %% [markdown]
# ## Setup 

# %%
seed = np.random.randint(0, 2**32 - 1)
warnings.filterwarnings("ignore")
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"

# %% [markdown]
# ## RL agent defintion

# %%
class PPOSurveillanceAgent(PPO):
  def __init__(self,
               env: gym.Env,
               # Look parameters
               azimuth_beamwidth: float,
               elevation_beamwidth: float,
               bandwidth: float,
               pulsewidth: float,
               prf: float,
               n_pulses: int,
               **kwargs):
    super().__init__(env=env, **kwargs)
    # Populate look parameters
    self.azimuth_beamwidth = azimuth_beamwidth
    self.elevation_beamwidth = elevation_beamwidth
    self.bandwidth = bandwidth
    self.pulsewidth = pulsewidth
    self.prf = prf
    self.n_pulses = n_pulses

    self.feature_net = nn.Sequential(
        nn.Conv2d(
            self.observation_space.shape[0], 32, kernel_size=8, stride=4),
        nn.ReLU(),
        nn.Conv2d(32, 64, kernel_size=4, stride=2),
        nn.ReLU(),
        nn.Conv2d(64, 64, kernel_size=3, stride=1),
        nn.ReLU(),
        nn.Flatten(start_dim=1, end_dim=-1),
        nn.LazyLinear(512),
        nn.ReLU(),
    )

    # The actor head parameterizes the mean and variance of a Gaussian distribution for the beam steering angles in az/el.
    n_continuous_actions = 2
    self.action_alpha = nn.Sequential(
        nn.Linear(512, n_continuous_actions),
        nn.Softplus(),
    )
    self.action_beta = nn.Sequential(
        nn.Linear(512, n_continuous_actions),
        nn.Softplus(),
    )

    self.critic = nn.Sequential(
        nn.Linear(512, 1),
    )

    self.save_hyperparameters()

  def forward(self, x: torch.Tensor):
    features = self.feature_net(x)
    # Sample the action from its distribution
    alpha = self.action_alpha(features) + 1
    beta = self.action_beta(features) + 1
    dist = torch.distributions.Beta(alpha, beta)
    actions = dist.sample()
    # Compute the value of the state
    values = self.critic(features).flatten()
    return actions, values

  def evaluate_actions(self,
                       observations: torch.Tensor,
                       actions: torch.Tensor):
    features = self.feature_net(observations)
    # Compute action sampling distribution
    alpha = self.action_alpha(features) + 1
    beta = self.action_beta(features) + 1
    dist = torch.distributions.Beta(alpha, beta)
    log_prob = dist.log_prob(actions)
    entropy = dist.entropy()
    return log_prob, entropy


  def configure_optimizers(self):
    optimizer = torch.optim.Adam(self.parameters(), lr=2.5e-4, eps=1e-5)
    return optimizer

  def on_train_epoch_end(self) -> None:
    if self.env.return_queue and self.env.length_queue:
      self.log_dict({
          'train/mean_episode_reward': np.mean(self.env.return_queue),
          'train/mean_episode_length': np.mean(self.env.length_queue),
          'train/total_step_count': float(self.total_step_count),
      },
          prog_bar=True, logger=True)



# %% [markdown]
# ## Environment setup

# %%
# In this experiment, targets move according to a constant velocity, white noise acceleration model.
# http://www.control.isy.liu.se/student/graduate/targettracking/file/le2_handout.pdf
transition_model = lambda state, dt: constant_velocity(state, q=10, dt=dt)

# Radar system object
radar = PhasedArrayRadar(
    ndim_state=6,
    position_mapping=(0, 2, 4),
    velocity_mapping=(1, 3, 5),
    position=np.array([[0], [0], [0]]),
    rotation_offset=np.array([[0], [0], [0]]),
    # Array parameters
    n_elements_x=32,
    n_elements_y=32,
    element_spacing=0.5,  # Wavelengths
    element_tx_power=10,
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
    state_vector=[  -40,   0,  20,   0, 10e3, 0],
    covar=np.diag([5, 100, 5, 100,  5e3, 100])
)

# Radar parameters
# Specifying these up here because they should really be part of the action space, but that will require some refactoring of my lightning-rl repo
# az_bw = el_bw = 3
az_bw = 3
el_bw = 3
bw = 100e6
pulsewidth = 10e-6
prf = 5e3
n_pulses = 32

# Create the environment
env = gym.make('mpar_sim/SimpleParticleSurveillance-v0',
               radar=radar,
               # Radar parameters
               azimuth_beamwidth=az_bw,
               elevation_beamwidth=el_bw,
               bandwidth=bw,
               pulsewidth=pulsewidth,
               prf=prf,
               n_pulses=n_pulses,
               transition_model=transition_model,
               initial_state=initial_state,
               birth_rate=0.025,
               death_probability=0.01,
               initial_number_targets=25,
               max_initial_number_targets=50,
               n_confirm_detections=3,
               randomize_initial_state=False,
               max_random_az_covar=0.1,
               max_random_el_covar=0.1,
               render_mode='rgb_array',
               )

# Define all environment wrappers
max_episode_steps = 500
# Observation wrappers
env = gym.wrappers.ResizeObservation(env, (64, 64))
env = ImageToPytorch(env)
env = SqueezeImage(env)
env = gym.wrappers.FrameStack(env, 4)
env = gym.wrappers.TimeLimit(env, max_episode_steps=max_episode_steps)

n_env = 32
env = gym.vector.AsyncVectorEnv([lambda: env]*n_env)
# n_env = 1
# env = gym.vector.SyncVectorEnv([lambda: env])

env = gym.wrappers.RescaleAction(env, 
                                 np.zeros(env.action_space.shape), 
                                 np.ones(env.action_space.shape))
env = gym.wrappers.RecordEpisodeStatistics(env=env, deque_size=20)


# %% [markdown]
# ## Training loop

# %%
# ppo_agent = PPOSurveillanceAgent(env,
#                                  n_rollouts_per_epoch=1,
#                                  n_steps_per_rollout=512,
#                                  n_gradient_steps=5,
#                                  batch_size=4096,
#                                  gamma=0.99,
#                                  gae_lambda=0.95,
#                                  value_coef=1,
#                                  entropy_coef=0.01,
#                                  seed=seed,
#                                  normalize_advantage=True,
#                                  policy_clip_range=0.2,
#                                  target_kl=None,
#                                  # Radar parameters
#                                  azimuth_beamwidth=az_bw,
#                                  elevation_beamwidth=el_bw,
#                                  bandwidth=bw,
#                                  pulsewidth=pulsewidth,
#                                  prf=prf,
#                                  n_pulses=n_pulses,
#                                  )

# trainer = pl.Trainer(
#     max_time="00:00:30:00",
#     gradient_clip_val=0.5,
#     accelerator='gpu',
#     devices=1,
# )
# trainer.fit(ppo_agent)


# %% [markdown]
# ## Create agents to be used for comparison

# %%
# TODO: Here, the action space is mapped from [0,1] to the az/el limits. When I designed this object, I did not have that in mind, so this is kinda hacky.
raster_agent = RasterScanAgent(
    azimuth_scan_limits=np.array([0, 1]),
    elevation_scan_limits=np.array([0, 1]),
    azimuth_beam_spacing=0.75,
    elevation_beam_spacing=0.75,
    azimuth_beamwidth=az_bw/(radar.az_fov[1] - radar.az_fov[0]),
    elevation_beamwidth=el_bw/(radar.el_fov[1] - radar.el_fov[0]),
    bandwidth=bw,
    pulsewidth=pulsewidth,
    prf=prf,
    n_pulses=n_pulses,
)
raster_agent.azimuth_beamwidth = az_bw
raster_agent.elevation_beamwidth = el_bw

checkpoint_filename = "/home/shane/src/mpar-sim/lightning_logs/version_93/checkpoints/epoch=156-step=3121.ckpt"
ppo_agent = PPOSurveillanceAgent.load_from_checkpoint(
    checkpoint_filename, env=env, seed=seed)
ppo_agent.eval()


# %% [markdown]
# ## Simulate each agent

# %%
ppo_init_ratio = np.ones((max_episode_steps, n_env))
ppo_tracks_init = np.zeros((max_episode_steps, n_env))
raster_init_ratio = np.ones((max_episode_steps, n_env))
raster_tracks_init = np.zeros((max_episode_steps, n_env))
beam_coverage_map = np.zeros((64, 64))

az_axis = np.linspace(radar.az_fov[0], radar.az_fov[1], beam_coverage_map.shape[1])
el_axis = np.linspace(radar.el_fov[0], radar.el_fov[1], beam_coverage_map.shape[0])


tic = time.time()
# Test the PPO agent

obs, info = env.reset(seed=seed)
dones = np.zeros(n_env, dtype=bool)
i = 0
with torch.no_grad():
  while not np.all(dones):
    obs_tensor = torch.as_tensor(obs).to(
        device=ppo_agent.device, dtype=torch.float32)
    action_tensor = ppo_agent.forward(obs_tensor)[0]
    actions = action_tensor.cpu().numpy()
    # Repeat actions for all environments
    obs, reward, terminated, truncated, info = env.step(actions)
    dones = np.logical_or(dones, np.logical_or(terminated, truncated))
    
    ppo_init_ratio[i, ~dones] = info['initiation_ratio'][~dones]
    ppo_tracks_init[i:, :] = info['n_tracks_initiated'][:]
    
    # Add 1 to the pixels illuminated by the current beam using np.digitize
    az_deg = -45 + 90 * actions[0,0]
    el_deg = -45 + 90 * actions[0,1]
    az = np.digitize(az_deg, az_axis)
    el = np.digitize(el_deg, el_axis[::-1])
    # beam_coverage_map *= 0.99
    beam_coverage_map[el, az] += 1
    
    i += 1
toc = time.time()
print("PPO agent done")
print(f"Time elapsed: {toc-tic:.2f} seconds")

# Test the raster agent
tic = time.time()

obs, info = env.reset(seed=seed)
dones = np.zeros(n_env, dtype=bool)
i = 0
while not np.all(dones):
  look = raster_agent.act(obs)
  actions = np.array([[look.azimuth_steering_angle,
                       look.elevation_steering_angle]])
  actions = np.repeat(actions, n_env, axis=0)
  # Repeat actions for all environments
  obs, reward, terminated, truncated, info = env.step(actions)
  dones = np.logical_or(dones, np.logical_or(terminated, truncated))
  raster_init_ratio[i, ~dones] = info['initiation_ratio'][~dones]
  raster_tracks_init[i:, :] = info['n_tracks_initiated'][:]
  i += 1
  
toc = time.time()
print("Raster agent done")
print(f"Time elapsed: {toc-tic:.2f} seconds")
env.close()

# %% [markdown]
# Plot the results

# %%
fig, ax = plt.subplots()
plt.plot(np.mean(raster_init_ratio[:-1, :], axis=1), linewidth=2)
plt.plot(np.mean(ppo_init_ratio[:-1, :], axis=1), linewidth=2)
plt.grid()
plt.xlabel('Time step (dwells)', fontsize=14)
plt.ylabel('Fraction of Targets Detected', fontsize=14)
plt.legend(['Raster', 'RL'], fontsize=14)

plt.figure()
plt.plot(np.mean(raster_tracks_init[:-1, :], axis=1), linewidth=2)
plt.plot(np.mean(ppo_tracks_init[:-1, :], axis=1), linewidth=2)
plt.xlabel('Time step (dwells)', fontsize=14)
plt.ylabel('Number of Tracks Initiated', fontsize=14)
plt.legend(['Raster', 'RL'], fontsize=14)
plt.grid()

plt.figure()
plt.imshow(beam_coverage_map,
           norm='linear',
           extent=[az_axis[0], az_axis[-1], el_axis[0], el_axis[-1]])


# # %%
# # Visualizations

# d = Designer(limits=[(-45, 45), (-45, 45)],
#              label=['azimuth (deg.)', 'elevation (deg.)'])
# animation = plot_contour(pos_history=env.swarm_optim.pos_history[::2],
#                          designer=d,)
# # animation.save('/home/shane/particles.gif', writer='ffmpeg',
# #               fps=10)
# # %%


# plotter = Plotter(Dimension.THREE)
# plotter.plot_sensors(radar, "Radar")
# plotter.plot_ground_truths(test_env.target_history, radar.position_mapping)
# plotter.plot_measurements(test_env.detection_history, radar.position_mapping)

plt.show()
env.close()


# %%



