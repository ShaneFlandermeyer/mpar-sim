# %% [markdown]
# ## Simple Particle Surveillance Environment

# %% [markdown]
# ## Imports

# %%
import time
import warnings

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
from lightning_rl.models.on_policy_models.ppg import PPG
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
             max_episode_steps=500):
  def thunk():
    # In this experiment, targets move according to a constant velocity, white noise acceleration model.
    # http://www.control.isy.liu.se/student/graduate/targettracking/file/le2_handout.pdf
    transition_model = ConstantVelocity(ndim_pos=3, noise_diff_coeff=10)
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
        state_vector=[30,   0,  30,   0, 10e3, 0],
        covar=np.diag([3**2, 100**2, 3**2, 100**2,  1000**2, 100**2])
    )

    env = gym.make(env_id,
                   radar=radar,
                   transition_model=transition_model,
                   initial_state=initial_state,
                   birth_rate=0.05,
                   death_probability=0.005,
                   initial_number_targets=50,
                   n_confirm_detections=2,
                   randomize_initial_state=True,
                   max_random_az_covar=7**2,
                   max_random_el_covar=7**2,
                   render_mode='rgb_array',
                   )

    # Wrappers
    # TODO: The ResizeObservation wrapper may have been breaking my stuff!!!
    # env = gym.wrappers.ResizeObservation(env, (64, 64))
    env = SqueezeImage(env)
    env = gym.wrappers.FrameStack(env, 4)
    env = gym.wrappers.TimeLimit(env, max_episode_steps=max_episode_steps)
    env = gym.wrappers.ClipAction(env)

    return env

  return thunk


# %% [markdown]
# ## RL agent defintion

# %%


class PPGSurveillanceAgent(PPG):
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
        ortho_init(nn.Conv2d(
            self.observation_space.shape[0], 32, kernel_size=8, stride=4)),
        nn.ReLU(),
        ortho_init(nn.Conv2d(32, 64, kernel_size=4, stride=2)),
        nn.ReLU(),
        ortho_init(nn.Conv2d(64, 64, kernel_size=3, stride=1)),
        nn.ReLU(),
        nn.Flatten(start_dim=1, end_dim=-1),
        ortho_init(nn.Linear(1024, 512)),
        nn.ReLU(),
    )

    # The actor head parameterizes the mean and variance of a Gaussian distribution for the beam steering angles in az/el.
    self.n_stochastic_actions = 2
    # self.actor = ortho_init(nn.Linear(512, self.n_stochastic_actions), std=0.01)
    self.action_mean = ortho_init(
        nn.Linear(512, self.n_stochastic_actions), std=0.01)
    self.action_std = nn.Sequential(
        ortho_init(nn.Linear(512, self.n_stochastic_actions), std=1.0),
        nn.Softplus(),
    )
    self.aux_critic = ortho_init(nn.Linear(512, 1), std=1)
    self.critic = ortho_init(nn.Linear(512, 1), std=1)
    self.save_hyperparameters()

  def forward(self, x: torch.Tensor):
    feature = self.feature_net(x / 255.0)
    # Sample the action from its distribution
    mean = self.action_mean(feature)
    std = self.action_std(feature)

    # Compute the value of the state
    value = self.critic(feature).flatten()
    aux_value = self.aux_critic(feature).flatten()
    return (mean, std), aux_value, value

  def act(self, observations: torch.Tensor):
    action_logits, value, aux_value = self.forward(observations)
    mean, std = action_logits
    action_dist = torch.distributions.Normal(mean, std)
    stochastic_actions = action_dist.sample()
    deterministic_actions = (
        # Azimuth beamwidth
        torch.full((observations.shape[0], 1), self.azimuth_beamwidth).to(
            stochastic_actions.device),
        # Elevation beamwidth
        torch.full((observations.shape[0], 1), self.elevation_beamwidth).to(
            stochastic_actions.device),
        # Bandwidth
        torch.full((observations.shape[0], 1), self.bandwidth).to(
            stochastic_actions.device),
        # Pulsewidth
        torch.full((observations.shape[0], 1), self.pulsewidth).to(
            stochastic_actions.device),
        # PRF
        torch.full((observations.shape[0], 1), self.prf).to(
            stochastic_actions.device),
        # Number of pulses
        torch.full((observations.shape[0], 1), self.n_pulses).to(
            stochastic_actions.device),
    )
    action = torch.cat((stochastic_actions,) + deterministic_actions, 1)
    return action, value, action_dist.log_prob(stochastic_actions), action_dist.entropy(), action_dist, aux_value

  def logits_to_action_dist(self, logits: torch.Tensor) -> torch.distributions.Distribution:
    """
    Return the action distribution from the output logits. This is needed to compute the KL divergence term in the joint loss on page 3 of the PPG paper (Cobbe2020).

    Parameters
    ----------
    logits : torch.Tensor
        Logits from the output of the actor network

    Returns
    -------
    torch.distributions.Distribution
        Action distribution
    """
    mean = logits[:, 0, :]
    std = logits[:, 1, :]
    action_dist = torch.distributions.Normal(mean, std)
    return action_dist

  def evaluate_actions(self,
                       observations: torch.Tensor,
                       actions: torch.Tensor):
    # Only evaluate stochastic actions
    actions = actions[:, :self.n_stochastic_actions]
    action_logits, aux_value, value = self.forward(observations)
    mean, std = action_logits
    action_dist = torch.distributions.Normal(mean, std)
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
max_episode_steps = 500
env = gym.vector.AsyncVectorEnv(
    [make_env(env_id,  max_episode_steps) for _ in range(n_env)])
env = gym.wrappers.RecordEpisodeStatistics(env=env, deque_size=20)


# %% [markdown]
# ## Training loop

# %%
az_bw = 3
el_bw = 3
bw = 100e6
pulsewidth = 10e-6
prf = 5e3
n_pulses = 32

# Compute policy/aux minibatch sizes
# The aux batch can get pretty large. My little baby 8 gb 3070 can't handle any larger than this
n_steps_per_rollout = 256
n_policy_steps = 32
n_policy_minibatch = 16
n_aux_minibatch = 64
policy_minibatch_size = n_steps_per_rollout*n_env//n_policy_minibatch
aux_minibatch_size = n_policy_steps*n_env//n_aux_minibatch

ppg_agent = PPGSurveillanceAgent(env,
                                 n_rollouts_per_epoch=1,
                                 n_steps_per_rollout=n_steps_per_rollout,
                                 shared_arch=True,
                                 gamma=0.99,
                                 gae_lambda=0.95,
                                 value_coef=1,
                                 entropy_coef=0,
                                 seed=seed,
                                 normalize_advantage=True,
                                 policy_clip_range=0.2,
                                 policy_minibatch_size=policy_minibatch_size,
                                 # PPG parameters
                                 aux_minibatch_size=aux_minibatch_size,
                                 n_policy_steps=n_policy_steps,
                                 n_policy_epochs=3,
                                 n_value_epochs=3,
                                 n_aux_epochs=6,
                                 beta_clone=1.0,
                                 # Radar parameters
                                 azimuth_beamwidth=az_bw,
                                 elevation_beamwidth=el_bw,
                                 bandwidth=bw,
                                 pulsewidth=pulsewidth,
                                 prf=prf,
                                 n_pulses=n_pulses,
                                 )

# checkpoint_filename = "/home/shane/src/mpar-sim/lightning_logs/version_481/checkpoints/epoch=31-step=61440.ckpt"
# ppg_agent = PPGSurveillanceAgent.load_from_checkpoint(
#     checkpoint_filename, env=env, seed=seed)

trainer = pl.Trainer(
    max_epochs=50,
    gradient_clip_val=0.5,
    accelerator='gpu',
    devices=1,
)
trainer.fit(ppg_agent)


# %% [markdown]
# ## Create agents to be used for comparison

# %%
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


# %% [markdown]
# ## Simulate each agent

# %%
ppo_init_ratio = np.ones((max_episode_steps, n_env))
ppo_tracks_init = np.zeros((max_episode_steps, n_env))
raster_init_ratio = np.ones((max_episode_steps, n_env))
raster_tracks_init = np.zeros((max_episode_steps, n_env))
beam_coverage_map = np.zeros((32, 32))

az_axis = np.linspace(-45, 45, beam_coverage_map.shape[1])
el_axis = np.linspace(-45, 45, beam_coverage_map.shape[0])



# Test the PPO agent
tic = time.time()
obs, info = env.reset(seed=seed)
dones = np.zeros(n_env, dtype=bool)
i = 0
with torch.no_grad():
  while not np.all(dones):
    obs_tensor = torch.as_tensor(obs).to(
        device=ppg_agent.device, dtype=torch.float32)
    action_tensor = ppg_agent.act(obs_tensor)[0]
    actions = action_tensor.cpu().numpy()
    # Repeat actions for all environments
    obs, reward, terminated, truncated, info = env.step(actions)
    dones = np.logical_or(dones, np.logical_or(terminated, truncated))

    ppo_init_ratio[i, ~dones] = info['initiation_ratio'][~dones]
    ppo_tracks_init[i:, ~np.logical_or(
        terminated, truncated)] = info['n_tracks_initiated'][~np.logical_or(terminated, truncated)]
    # if i == 100:
    #   plt.imshow(obs[0, 3, :, :])
    #   plt.show()

    # Add 1 to the pixels illuminated by the current beam using np.digitize
    # if i > 100 and not dones[0]:
    actions[0, :] = wrap_to_interval(actions[0, :], -45, 45)
    az = np.digitize(actions[0, 0], az_axis, right=True)
    el = np.digitize(actions[0, 1], el_axis[::-1], right=True)
    beam_coverage_map[max(el-2, 0):min(el+2, len(el_axis)), max(az-2, 0):min(az+2, len(az_axis))] += 1
    # beam_coverage_map *= 0.99

    i += 1
toc = time.time()
print("PPO agent done")
print(f"Time elapsed: {toc-tic:.2f} seconds")

# # Test the raster agent
tic = time.time()

obs, info = env.reset(seed=seed)
dones = np.zeros(n_env, dtype=bool)
i = 0
while not np.all(dones):
  look = raster_agent.act(obs)
  actions = np.array([[look.azimuth_steering_angle,
                     look.elevation_steering_angle,
                     look.azimuth_beamwidth,
                     look.elevation_beamwidth,
                     look.bandwidth,
                     look.pulsewidth,
                     look.prf,
                     look.n_pulses]])
  actions = np.repeat(actions, n_env, axis=0)
  # Repeat actions for all environments
  obs, reward, terminated, truncated, info = env.step(actions)
  dones = np.logical_or(dones, np.logical_or(terminated, truncated))
  raster_init_ratio[i, ~dones] = info['initiation_ratio'][~dones]
  raster_tracks_init[i:, ~np.logical_or(
      terminated, truncated)] = info['n_tracks_initiated'][~np.logical_or(terminated, truncated)]
#   if i == 200:
#       plt.imshow(obs[0, 3, :, :])
#       plt.show()
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
plt.ylabel('Track Initiation Fraction', fontsize=14)
plt.legend(['Raster', 'RL'], fontsize=14)

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
