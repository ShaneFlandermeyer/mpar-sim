# %%
# Imports
import time

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

import mpar_sim.envs
from mpar_sim.defaults import (default_gbest_pso, default_lbest_pso,
                               default_radar, default_raster_scan_agent,
                               default_scheduler)
from mpar_sim.wrappers.image_to_pytorch import ImageToPytorch
from mpar_sim.wrappers.squeeze_image import SqueezeImage
import pytorch_lightning as pl
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
    ConstantVelocity(10),
    ConstantVelocity(10),
    ConstantVelocity(10),
])

# NOTE: Specifying initial state in terms of az/el/range (in degrees)!
initial_state = GaussianState(
    state_vector=[0, 0, 0, 0, 15e3, 0],
    covar=np.diag([25, 100, 25, 100, 5e3, 100])
)
# Radar system object
radar = default_radar()
radar.false_alarm_rate = 1e-7
radar.include_false_alarms = False
radar.element_tx_power = 1000
radar.max_range = 20e3
scheduler = default_scheduler(radar)

# Environment creation
env = gym.make('mpar_sim/ParticleSurveillance-v0',
               radar=radar,
               transition_model=transition_model,
               initial_state=initial_state,
               birth_rate=0,
               death_probability=0,
               initial_number_targets=25,
               n_confirm_detections=3,
               randomize_initial_state=True,
               render_mode='rgb_array',
               )
# Wrap the environment


env = gym.wrappers.ResizeObservation(env, (84, 84))
env = ImageToPytorch(env)
env = SqueezeImage(env)
env = gym.wrappers.FrameStack(env, 4)

n_env = 8
env = gym.vector.AsyncVectorEnv([lambda: env]*n_env)
env = gym.wrappers.RecordEpisodeStatistics(env=env, deque_size=20)

# %%
# Create the agent and run the simulation
# agent = default_raster_scan_agent()
seed = np.random.randint(0, 2**32 - 1)
ppo_agent = PPOSurveillanceAgent(env,
                                 n_rollouts_per_epoch=10,
                                 n_steps_per_rollout=128,
                                 n_gradient_steps=3,
                                 batch_size=256,
                                 gamma=0.99,
                                 gae_lambda=0.9,
                                 value_coef=0.5,
                                 entropy_coef=0.01,
                                 seed=seed,
                                 normalize_advantage=True,
                                 policy_clip_range=0.1,
                                 # Radar parameters
                                 azimuth_beamwidth=5,
                                 elevation_beamwidth=5,
                                 bandwidth=100e6,
                                 pulsewidth=10e-6,
                                 prf=5e3,
                                 n_pulses=100,
                                 )

trainer = pl.Trainer(
    max_time="00:03:00:00",
    gradient_clip_val=0.5,
    accelerator='gpu',
    devices=1,
)
trainer.fit(ppo_agent)

# obs, info = env.reset()
# tic = time.time()
# i = 0
# while True:
#   i += 1
#   # Create a look and schedule it. This fills in the tx power field based on the number of elements used to form the beam
#   action = agent.act(env.time)
#   scheduler.schedule(list(action), env.time)
#   look = scheduler.manager.allocated_tasks.pop()

#   obs, reward, terminated, truncated, info = env.step(
#       dict(
#           azimuth_steering_angle=look.azimuth_steering_angle,
#           elevation_steering_angle=look.elevation_steering_angle,
#           azimuth_beamwidth=agent.azimuth_beamwidth,
#           elevation_beamwidth=agent.elevation_beamwidth,
#           bandwidth=agent.bandwidth,
#           pulsewidth=agent.pulsewidth,
#           prf=agent.prf,
#           n_pulses=agent.n_pulses,
#           tx_power=look.tx_power,
#       )

#   )
#   done = terminated or truncated
#   if done:
#     # At this point, you would normally reset the environment. For this demonstration, we just break out of the loop
#     toc = time.time()
#     print("Episode finished after {} timesteps".format(i))
#     print(f"Episode took {toc-tic} seconds")
#     env.close()
#     break


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
# plotter.plot_ground_truths(env.target_history, radar.position_mapping)
# plotter.plot_measurements(env.detection_history, radar.position_mapping)
# plt.show()
# # %%
