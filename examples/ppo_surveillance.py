# %%
# Imports
import copy
import time
import cv2

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
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
import torch
from mpar_sim.agents.raster_scan import RasterScanAgent
from mpar_sim.beam.beam import GaussianBeam, RectangularBeam
from mpar_sim.common.wrap_to_interval import wrap_to_interval

import mpar_sim.envs
from mpar_sim.defaults import (default_gbest_pso, default_lbest_pso,
                               default_radar, default_raster_scan_agent,
                               default_scheduler)
from mpar_sim.radar import PhasedArrayRadar
from mpar_sim.wrappers.image_to_pytorch import ImageToPytorch
from mpar_sim.wrappers.squeeze_image import SqueezeImage
import pytorch_lightning as pl

import warnings
warnings.filterwarnings("ignore")
# %%
# Agent object definition


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
    self.action_mean = nn.Sequential(
        nn.Linear(512, n_continuous_actions),
    )

    self.action_variance = nn.Sequential(
        nn.Linear(512, n_continuous_actions),
        nn.Softplus(),
    )

    self.critic = nn.Sequential(
        nn.Linear(512, 1),
    )

    self.save_hyperparameters()

  def forward(self, x: torch.Tensor):
    features = self.feature_net(x)
    # Sample actions from a Gaussian distribution
    means = self.action_mean(features)
    variances = self.action_variance(features)
    dist = torch.distributions.Normal(means, variances)
    return dist, self.critic(features).flatten()

  def predict(self, obs, deterministic: bool = True):
    features = self.feature_net(obs)
    mean = self.action_mean(features)
    if deterministic:
      return mean.detach().cpu().numpy()
    else:
      variances = self.action_variance(features)
      dist = torch.distributions.Normal(mean, variances)
      return dist.sample().detach().cpu().numpy()

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


# %%
# Set up the environment
# Target generation model
transition_model = CombinedLinearGaussianTransitionModel([
    ConstantVelocity(0),
    ConstantVelocity(0),
    ConstantVelocity(0),
])

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
    element_tx_power=20,
    # System parameters
    center_frequency=3e9,
    system_temperature=290,
    noise_figure=4,
    # Scan settings
    beam_shape=GaussianBeam,
    az_fov=[-45, 45],
    el_fov=[-45, 45],
    # Detection settings
    false_alarm_rate=1e-6,
    include_false_alarms=False
)

# NOTE: Specifying initial state in terms of az/el/range (in degrees)!
initial_state = GaussianState(
    state_vector=[np.random.uniform(-45, 45), 
                  np.random.uniform(-45, 45), 
                  0, 
                  0, 15e3, 0],
    covar=np.diag([10, 100, 10, 100, 1e3, 100])
)
print(initial_state.state_vector)


# Environment creation
env = gym.make('mpar_sim/ParticleSurveillance-v0',
               radar=radar,
               # Radar parameters
               azimuth_beamwidth  =5,
               elevation_beamwidth=5,
               bandwidth=100e6,
               pulsewidth=10e-6,
               prf=5e3,
               n_pulses=32,
               transition_model=transition_model,
               initial_state=initial_state,
               birth_rate=0,
               death_probability=0,
               initial_number_targets=25,
               n_confirm_detections=3,
               randomize_initial_state=False,
               max_random_az_covar=50,
               max_random_el_covar=50,
               render_mode='rgb_array',
               )
# Wrap the environment
n_env = 16
max_episode_steps = 2000

env = gym.wrappers.ResizeObservation(env, (84, 84))
env = ImageToPytorch(env)
env = SqueezeImage(env)
env = gym.wrappers.FrameStack(env, 4)
env = gym.wrappers.TimeLimit(env, max_episode_steps=max_episode_steps)
test_env = gym.vector.SyncVectorEnv([lambda: env])


# env = gym.vector.AsyncVectorEnv([lambda: env]*n_env)
# env = gym.wrappers.RecordEpisodeStatistics(env=env, deque_size=20)
seed = np.random.randint(0, 2**32 - 1)
# %%
# Train the RL agent
# ppo_agent = PPOSurveillanceAgent(env,
#                                  n_rollouts_per_epoch=10,
#                                  n_steps_per_rollout=128,
#                                  n_gradient_steps=3,
#                                  batch_size=256,
#                                  gamma=0.99,
#                                  gae_lambda=0.9,
#                                  value_coef=0.5,
#                                  entropy_coef=0.01,
#                                  seed=seed,
#                                  normalize_advantage=True,
#                                  policy_clip_range=0.1,
#                                  # Radar parameters
#                                  azimuth_beamwidth=5,
#                                  elevation_beamwidth=5,
#                                  bandwidth=100e6,
#                                  pulsewidth=10e-6,
#                                  prf=5e3,
#                                  n_pulses=100,
#                                  )

# trainer = pl.Trainer(
#     max_time="00:00:30:00",
#     gradient_clip_val=0.5,
#     accelerator='gpu',
#     devices=1,
# )
# trainer.fit(ppo_agent)

# %%
# Test the agents
# Create the agent and run the simulation
raster_agent = RasterScanAgent(
    azimuth_scan_limits=np.array([-45, 45]),
    elevation_scan_limits=np.array([-45, 45]),
    azimuth_beam_spacing=0.5,
    elevation_beam_spacing=0.5,
    azimuth_beamwidth  =5,
    elevation_beamwidth=5,
    bandwidth=100e6,
    pulsewidth=10e-6,
    prf=5e3,
    n_pulses=100,
)
# TODO: Break this back into two loops
checkpoint_filename = "/home/shane/src/mpar-sim/lightning_logs/version_11/checkpoints/epoch=199-step=23904.ckpt"
ppo_agent = PPOSurveillanceAgent.load_from_checkpoint(
    checkpoint_filename, env=test_env, seed=seed)
ppo_agent.eval()

# raster_init_ratio = np.zeros((max_episode_steps,))
ppo_init_ratio = np.ones((max_episode_steps,))
raster_init_ratio = np.ones((max_episode_steps,))
az_axis = np.linspace(radar.az_fov[0], radar.az_fov[1], 32)
el_axis = np.linspace(radar.el_fov[0], radar.el_fov[1], 32)
beam_coverage_map = np.zeros((len(el_axis), len(az_axis)))

obs, info = test_env.reset()
tic = time.time()
i = 0
done = False
while not done:
  look = raster_agent.act(obs)
  actions = np.array([[look.azimuth_steering_angle,
                     look.elevation_steering_angle]])
  obs, reward, terminated, truncated, info = test_env.step(actions.T)
  done = terminated or truncated
  # Compute the fraction of targets whose tracks have been initiated
  if not terminated:
    raster_init_ratio[i] = info['initiation_ratio']
  
  i += 1
toc = time.time()
print("Episode finished after {} timesteps".format(i))
print(f"Episode took {toc-tic} seconds")


obs, info = test_env.reset()
tic = time.time()
i = 0
done = False
while not done:
  obs_tensor = torch.as_tensor(obs).to(
      device=ppo_agent.device, dtype=torch.float32)
  action_tensor = ppo_agent.forward(obs_tensor)[0].sample()
  actions = action_tensor.detach().cpu().numpy()
  # When the variance is high, I've found that it's better to wrap the actions back into the valid range rather than clip them. Helps with exploration
  actions[0,0] = wrap_to_interval(actions[0,0], radar.az_fov[0], radar.az_fov[1])
  actions[0,1] = wrap_to_interval(actions[0,1], radar.el_fov[0], radar.el_fov[1])
  
  obs, reward, terminated, truncated, info = test_env.step(actions.T)
  done = terminated or truncated
  # Compute the fraction of targets whose tracks have been initiated
  if not terminated:
    ppo_init_ratio[i] = info['initiation_ratio']
      
#   if i > 500:
#     az_start = np.digitize(actions[0, 0] - 1.5, az_axis) - 1
#     el_start = np.digitize(actions[0, 1] - 1.5, el_axis) - 1
#     az_stop = np.digitize(actions[0, 0] +  1.5, az_axis) - 1
#     el_stop = np.digitize(actions[0, 1] +  1.5, el_axis) - 1
#     beam_coverage_map[el_start:el_stop, az_start:az_stop] += 1

  i += 1
toc = time.time()
print("Episode finished after {} timesteps".format(i))
print(f"Episode took {toc-tic} seconds")

plt.figure()
plt.plot(raster_init_ratio[:-1])
plt.plot(ppo_init_ratio[:-1])
plt.xlabel('Time step (dwells)')
plt.ylabel('Fraction of Targets Detected')
plt.legend(['Raster', 'RL'])

# plt.figure()
# plt.imshow(beam_coverage_map, 
#            norm='linear', 
#            extent=[az_axis[0], az_axis[-1], el_axis[0], el_axis[-1]])


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
