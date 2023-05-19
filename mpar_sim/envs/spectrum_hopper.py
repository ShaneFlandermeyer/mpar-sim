import argparse
import itertools
import os
from collections import deque

import cv2
import gymnasium as gym
import numpy as np
import pygame
import ray
from ray import tune
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.models import ModelCatalog
from ray.tune.logger import pretty_print

from mpar_sim.interference.recorded import RecordedInterference
from mpar_sim.interference.hopping import HoppingInterference
from mpar_sim.models.networks.rnn import LSTMActorCritic


class SpectrumHopper(gym.Env):
  metadata = {"render.modes": ["rgb_array", "human"], "render_fps": 30}

  def __init__(self, config):
    super().__init__()
    # Environment config
    self.max_steps = config.get("max_steps", 100)
    self.cpi_len = config.get("cpi_len", 32)
    self.fft_size = config.get("fft_size", 1024)
    self.pri = config.get("pri", 10)
    self.gamma_state = config.get("gamma_state", 0.8)
    self.beta_bw = config.get("beta_fc", 0.0)
    self.beta_fc = config.get("beta_fc", 0.0)

    # Number of collisions that can be tolerated for reward function
    self.min_bandwidth = config.get("min_bandwidth", 0)
    self.min_collision_bw = config.get("min_collision_bw", 0)
    self.max_collision_bw = config.get("max_collision_bw", 1)

    self.freq_axis = np.linspace(0, 1, self.fft_size)
    # self.interference = HoppingInterference(
    #     start_freq=np.min(self.freq_axis),
    #     bandwidth=0.2,
    #     duration=self.pri,
    #     hop_size=0.2,
    #     channel_bw=1,
    #     fft_size=self.fft_size
    # )
    self.interference = RecordedInterference(
        "/home/shane/data/HOCAE_Snaps_bool_cleaned.dat", self.fft_size)

    self.max_obs = np.sum(self.gamma_state**np.arange(self.pri))
    self.observation_space = gym.spaces.Box(
        low=0.0, high=1.0, shape=(self.fft_size,))
    self.action_space = gym.spaces.Box(
        low=np.array([0.0, self.min_bandwidth]),
        high=np.array([1-self.min_bandwidth, 1]))

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
    self.pulse_count = 0
    self.history_len = self.cpi_len*self.pri
    self.history = {
        "radar": deque([np.zeros(self.fft_size, dtype=np.uint8) for _ in range(self.history_len)], maxlen=self.history_len),
        "interference": deque([np.zeros(self.fft_size, dtype=np.uint8) for _ in range(self.history_len)], maxlen=self.history_len),
        "bandwidth": deque([0 for _ in range(self.cpi_len)], maxlen=self.cpi_len),
        "center_freq": deque([0 for _ in range(self.cpi_len)], maxlen=self.cpi_len),
    }

    self.interference.reset()
    self.n_shift = self.np_random.integers(-self.fft_size//4, self.fft_size//4)
    self.interference.state = np.roll(self.interference.state, self.n_shift)

    obs = np.zeros(self.observation_space.shape)
    for _ in range(self.pri):
      self.interference.step(self.time)
      # TODO: Uncomment this for recorded interference
      # self.interference.state = np.roll(self.interference.state, self.n_shift)
      obs = self.interference.state + self.gamma_state*obs
      self.time += 1
      self.history["radar"].append(np.zeros_like(self.interference.state))
      self.history["interference"].append(self.interference.state)
    obs /= self.max_obs

    info = {}
    return obs, info

  def step(self, action):
    obs = np.zeros(self.observation_space.shape)
    for i in range(self.pri):
      # TODO: Uncomment for SAA agent. There's definitely a way to define this policy in RLLib
      # widest = self._get_widest(self.interference.state)
      # action = widest / self.fft_size
      self.interference.step(self.time)
      # TODO: Uncomment this for recorded interference
      # self.interference.state = np.roll(self.interference.state, self.n_shift)
      obs = self.interference.state + self.gamma_state*obs
      self.time += 1
      if i == 0:
        reward, radar_spectrum, info = self._get_reward(action)
      else:
        radar_spectrum = np.zeros_like(self.interference.state)
      self.history["radar"].append(radar_spectrum)
      self.history["interference"].append(self.interference.state)
    obs /= self.max_obs

    self.pulse_count += 1
    terminated = False
    truncated = self.pulse_count >= self.max_steps

    if self.render_mode == "human":
      self._render_frame()

    return obs, reward, terminated, truncated, info

  def render(self):
    if self.render_mode == "rgb_array":
      return self._render_frame()

  def _get_reward(self, action):
    # Compute radar spectrum
    start_freq = action[0]
    stop_freq = action[1]
    center_freq = 0.5*(start_freq + stop_freq)
    bandwidth = stop_freq - start_freq
    radar_spectrum = np.logical_and(
        self.freq_axis >= start_freq,
        self.freq_axis <= stop_freq)
    n_radar_bins = np.count_nonzero(radar_spectrum)

    # Compute spectrum-based rewards
    n_collisions = np.count_nonzero(np.logical_and(
        radar_spectrum, self.interference.state))
    r_spectrum = bandwidth
    if n_collisions > self.min_collision_bw*n_radar_bins:
      r_spectrum *= 1 - n_collisions / (self.max_collision_bw*n_radar_bins)

    # Compute distorition-based rewards
    self.history["bandwidth"].append(bandwidth)
    self.history["center_freq"].append(center_freq)
    r_distortion = self.beta_bw * \
        np.std(self.history["bandwidth"]) + self.beta_fc * \
        np.std(self.history["center_freq"])

    reward = r_spectrum - r_distortion

    # Reset per-CPI metrics
    if len(self.history["bandwidth"]) == self.cpi_len:
      self.history["bandwidth"].clear()
      self.history["center_freq"].clear()

    # Evaluation metrics
    widest = self._get_widest(self.interference.state)
    widest_bw = np.clip((widest[1] - widest[0]) / self.fft_size,
                        1/self.fft_size, None)

    info = {
        'bandwidth': max(bandwidth, 0),
        'missed': widest_bw - bandwidth,
        'collision': n_collisions / n_radar_bins if n_radar_bins > 0 else 0,
    }

    return reward, radar_spectrum, info

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
    intersection = np.logical_and(radar_spectrogram.T, spectrogram.T)

    pixels = spectrogram.T*100
    pixels[radar_spectrogram.T == 1] = 255
    pixels[intersection] = 150
    pixels = cv2.resize(pixels.astype(np.float32),
                        self.window_size, interpolation=cv2.INTER_AREA)

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
