import argparse
import itertools
import os

import gymnasium as gym
import numpy as np
import pygame
import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.pg import PGConfig
from ray.tune.logger import pretty_print

from mpar_sim.interference.hopping import HoppingInterference
from mpar_sim.interference.recorded import RecordedInterference
from mpar_sim.interference.single_tone import SingleToneInterference
from collections import deque
import cv2
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.models import ModelCatalog
from ray import air, tune

# from mpar_sim.models.networks.rllib_rnn import TorchLSTMModel


class SpectrumHopper(gym.Env):
  metadata = {"render.modes": ["rgb_array", "human"], "render_fps": 60}

  def __init__(self, config):
    super().__init__()
    # Environment config
    self.max_steps = config.get("max_steps", 100)
    self.history_len = config.get("history_len", 256)
    self.fft_size = config.get("fft_size", 1024)

    # Number of collisions that can be tolerated for reward function
    self.min_collisions = config.get("min_collisions", 0)
    self.max_collisions = config.get("max_collisions", self.fft_size)

    self.freq_axis = np.linspace(0, 1, self.fft_size)
    # self.interference = HoppingInterference(
    #     start_freq=np.min(self.freq_axis),
    #     bandwidth=0.2,
    #     duration=5,
    #     hop_size=0.2,
    #     channel_bw=1,
    #     fft_size=self.fft_size
    # )
    self.interference = RecordedInterference(
        "/home/shane/data/HOCAE_Snaps_bool_cleaned.dat", self.fft_size)
    self.observation_space = gym.spaces.Box(
        low=0.0, high=1.0, shape=(self.fft_size,))
    self.action_space = gym.spaces.Box(low=0.0, high=1.0, shape=(2,))

    # Render config
    render_mode = config.get("render_mode", "rgb_array")
    assert render_mode in self.metadata["render.modes"]
    self.render_mode = render_mode
    # Pygame setup
    self.window_size = config.get("window_size", (512, 512))
    self.window = None
    self.clock = None

  def reset(self, *, seed=None, options=None):
    self.time = 0
    self.interference.reset()
    self.history = {
        "radar": deque([np.zeros(self.fft_size, dtype=np.uint8) for _ in range(self.history_len)], maxlen=self.history_len),
        "interference": deque([np.zeros(self.fft_size, dtype=np.uint8) for _ in range(self.history_len)], maxlen=self.history_len),
    }
    obs = self.interference.state
    info = {}
    return obs, info

  def step(self, action):
    self.interference.step(self.time)
    reward = self._get_reward(action)
    self.time += 1
    terminated = False
    truncated = self.time >= self.max_steps
    obs = self.interference.state
    info = {}

    if self.render_mode == "human":
      self._render_frame()

    return obs, reward, terminated, truncated, info

  def render(self):
    if self.render_mode == "rgb_array":
      return self._render_frame()

  def _get_reward(self, action):
    # Compute radar spectrum
    start_freq = action[0]
    stop_freq = np.clip(action[1], start_freq, None)
    radar_bw = stop_freq - start_freq
    radar_spectrum = np.logical_and(
        self.freq_axis >= start_freq,
        self.freq_axis <= stop_freq)

    # Compute collision metrics
    n_collisions = np.count_nonzero(np.logical_and(
        radar_spectrum, self.interference.state))

    widest = self._get_widest(self.interference.state)
    widest_bw = np.clip((widest[1] - widest[0]) /
                        self.fft_size, 1/self.fft_size, None)
    # Reward is proportional to the radar bandwidth, and decreases linearly as the collision bandwidth increases to the maximum allowable value.
    # The result is clipped so that the reward is non-negative. This just makes analysis easier, and is not strictly necessary.
    reward = (radar_bw/widest_bw)
    if n_collisions > self.min_collisions:
      reward *= np.clip(
          1 - (n_collisions - self.min_collisions) / (self.max_collisions - self.min_collisions), 0, 1)
    # Update histories
    self.history["radar"].append(radar_spectrum.astype(np.uint8))
    self.history["interference"].append(
        self.interference.state.astype(np.uint8))

    return reward

  def _get_widest(self, spectrum: np.ndarray):
    # Group consecutive bins by value
    gap_widths = np.array([[x[0], len(list(x[1]))]
                          for x in itertools.groupby(spectrum)])
    vals = gap_widths[:, 0]
    widths = gap_widths[:, 1]
    starts = np.cumsum(widths) - widths

    # Compute the start frequency and width of the widest gap (in bins)
    open = (vals == 0)
    if not np.any(open):
      return np.array([0, 0])
    istart_widest = np.argmax(widths[open])
    widest_start = starts[open][istart_widest]
    widest_bw = widths[open][istart_widest]
    widest_stop = widest_start + widest_bw

    widest_start = widest_start
    widest_stop = widest_stop
    return np.array([widest_start, widest_stop])

  def _render_frame(self):
    if self.window is None and self.render_mode == "human":
      pygame.init()
      pygame.display.init()
      self.window = pygame.display.set_mode(self.window_size)

    if self.clock is None and self.render_mode == "human":
      self.clock = pygame.time.Clock()

    radar_spectrogram = np.stack(self.history["radar"], axis=0)
    spectrogram = np.stack(self.history["interference"], axis=0)

    pixels = spectrogram.T*100
    pixels[radar_spectrogram.T == 1] = 255
    pixels = cv2.resize(pixels, self.window_size, interpolation=cv2.INTER_AREA)

    if self.render_mode == "human":
      # Copy canvas drawings to the window
      canvas = pygame.surfarray.make_surface(pixels)
      self.window.blit(canvas, canvas.get_rect())
      pygame.event.pump()
      pygame.display.update()

      # Ensure that human rendering occurs at the pre-defined framerate
      self.clock.tick(self.metadata["render_fps"])
    else:
      return pixels

    pass


def get_cli_args():
  """Create CLI parser and return parsed arguments"""
  parser = argparse.ArgumentParser()

  # general args
  parser.add_argument("--num-cpus", type=int, default=12)
  parser.add_argument(
      "--framework",
      choices=["tf", "tf2", "torch"],
      default="torch",
      help="The DL framework specifier.",
  )
  parser.add_argument("--eager-tracing", action="store_true")
  parser.add_argument(
      "--stop-timesteps",
      type=int,
      default=2_000_000,
      help="Number of timesteps to train.",
  )
  parser.add_argument(
      "--stop-reward",
      type=float,
      default=4500.0,
      help="Reward at which we stop training.",
  )

  args = parser.parse_args()
  print(f"Running with following CLI args: {args}")
  return args


if __name__ == '__main__':
  args = get_cli_args()

  # ray.init(num_cpus=args.num_cpus or None)
  # ModelCatalog.register_custom_model("gru", TorchGRUModel)
  config = (
      PPOConfig()
      .environment(env=SpectrumHopper, normalize_actions=True,
                   env_config={"max_steps": 2500, "min_collisions": 5, "max_collisions": 10})
      .resources(
          num_gpus=int(os.environ.get("RLLIB_NUM_GPUS", "0")),
      )
      .training(
          train_batch_size=max(1024, 512*args.num_cpus),
          model={
              "fcnet_hiddens": [256, 256],
              
              # LSTM config
              # NOTE: This architecture plays basically optimallyk for the hopper once the first few steps are in the latent space.
              "use_lstm": True,
              "lstm_cell_size": 256,
              "lstm_use_prev_action": False,
              "lstm_use_prev_reward": False,
          },
          lr=2.5e-4,
          gamma=0.8,
          lambda_=0.95,
          clip_param=0.2,
          sgd_minibatch_size=512*args.num_cpus//4,
          num_sgd_iter=10,
          vf_loss_coeff=0.25,
      )
      .rollouts(num_rollout_workers=args.num_cpus, rollout_fragment_length="auto")
      .framework(args.framework, eager_tracing=args.eager_tracing)
  )
  
  

  # Training loop
  print("Configuration successful. Running training loop.")
  algo = config.build()
  highest_mean_reward = -np.inf
  while True:
    result = algo.train()
    print(pretty_print(result))
    # Save the best performing model found so far
    if result["episode_reward_mean"] > highest_mean_reward:
      highest_mean_reward = result["episode_reward_mean"]
      save_path = algo.save()
      
    if result["timesteps_total"] >= args.stop_timesteps or result["episode_reward_mean"] >= args.stop_reward:
      break

  # print("Finished training. Running manual test/inference loop.")
  if 'save_path' not in locals():
    save_path = "/home/shane/ray_results/PPO_SpectrumHopper_2023-05-09_15-17-14x7g9ev68/checkpoint_000215"
  algo = Algorithm.from_checkpoint(save_path)
  # Prepare env
  env_config = config["env_config"]
  env_config["render_mode"] = "human"
  env = SpectrumHopper(env_config)
  obs, info = env.reset()
  done = False
  total_reward = 0

  # Initialize memory
  lstm_cell_size = config["model"]["lstm_cell_size"]
  init_state = state = [
      np.zeros([lstm_cell_size], np.float32) for _ in range(2)]
  prev_action = np.zeros(env.action_space.shape, np.float32)
  prev_reward = 0
  while not done:
    # action = env.action_space.sample()
    action, state, _ = algo.compute_single_action(
        obs, state, prev_action=prev_action, prev_reward=prev_reward, explore=False)
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    prev_action = action
    prev_reward = reward

    total_reward += reward
    env.render()
  print("Total eval. reward:", total_reward)
