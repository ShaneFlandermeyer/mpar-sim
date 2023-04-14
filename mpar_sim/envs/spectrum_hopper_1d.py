import copy
from typing import List, Optional
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pygame
import cv2
import itertools


class SpectrumHopper1D(gym.Env):
  metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}
  """
  This gym environment formulates the interference avoidance problem as a continuous control task.
  
  Hopefully, the agent will be able to learn from "spectrograms" that have been pre-processed to form a binary mask indicating the presence of interference.
  """

  ##########################
  # Core Gym methods
  ##########################
  def __init__(self,
               filename: str,
               channel_bandwidth: float = 100e6,
               fft_size: int = 1024,
               n_image_snapshots: int = 512,
               render_mode: str = None) -> None:
    self.observation_space = gym.spaces.Box(
        low=0, high=1, shape=(20, ), dtype=np.float32)

    # Action space is the start and span of the radar waveform
    self.action_space = gym.spaces.Box(
        low=-1, high=1, shape=(2,), dtype=np.float32)

    self.filename = filename
    self.channel_bandwidth = channel_bandwidth
    self.fft_size = fft_size
    self.n_image_snapshots = n_image_snapshots
    self.render_mode = render_mode

    self.fft_freq_axis = np.linspace(
        -self.channel_bandwidth/2, self.channel_bandwidth/2, self.fft_size)
    self.data = np.fromfile(filename, dtype=np.uint8)
    n_snapshots = int(self.data.size / fft_size)
    self.data = self.data.reshape((n_snapshots, fft_size), order='C')

    # Rendering
    assert render_mode is None or render_mode in self.metadata["render_modes"]
    self.render_mode = render_mode
    self.window = None
    self.clock = None

  def step(self, action: np.ndarray):
    # Update the radar waveform
    start_freq, stop_freq = self._parse_action(action)
    self.radar_spectrogram = np.roll(self.radar_spectrogram, 1, axis=0)
    self.radar_spectrogram[0] = np.logical_and(
        self.fft_freq_axis >= start_freq,
        self.fft_freq_axis <= stop_freq)

    reward = self._compute_reward(
        self.radar_spectrogram[0], self.spectrogram[0])

    # Update communications spectrum
    self.spectrogram = np.roll(self.spectrogram, 1, axis=0)
    self.start_ind = (self.start_ind + 1) % self.data.shape[0]
    self.spectrogram[0] = self.data[self.start_ind]

    if self.render_mode == "human":
      self._render_frame()

    obs = self._get_obs()
    info = {}
    terminated = False
    truncated = False
    done = terminated or truncated
    return obs, reward, terminated, truncated, info

  def reset(self, seed: int = None, options: dict = None):
    super().reset(seed=seed)

    # Shift the data to get more diversity in the training
    # self.data = np.roll(self.data, self.np_random.integers(0, 512), axis=1)

    # Randomly sample spectrogram from a file
    self.start_ind = self.np_random.integers(
        0, self.data.shape[0] - self.n_image_snapshots)
    stop_ind = self.start_ind + self.n_image_snapshots
    self.spectrogram = self.data[self.start_ind:stop_ind]
    self.radar_spectrogram = np.zeros_like(self.spectrogram)

    obs = self._get_obs()
    info = {}

    if self.render_mode == "human":
      self._render_frame()

    return obs, info

  def render(self):
    if self.render_mode == "rgb_array":
      return self._render_frame()

  def close(self):
    if self.window is not None:
      pygame.display.quit()
      pygame.quit()

  ##########################
  # Helper methods
  ##########################

  def _parse_action(self, action):
    start_freq = action[0] * self.channel_bandwidth / 2
    stop_freq = np.clip(action[1] * self.channel_bandwidth / 2,
                        start_freq, None)
    return start_freq, stop_freq

  def _compute_reward(self,
                      radar_spectrum: np.ndarray,
                      interference: np.ndarray,
                      ):
    widest_start, widest_stop = self._get_widest(interference)
    n_widest = widest_stop - widest_start
    n_radar_bins = np.sum(radar_spectrum)
    n_collisions = np.sum(np.logical_and(radar_spectrum, interference))
    reward = n_radar_bins / n_widest if n_collisions < 10 else 0
    return reward

  def _get_obs(self):
    # Group consecutive bins by value
    gap_widths = np.array([[x[0], len(list(x[1]))]
                          for x in itertools.groupby(self.spectrogram[0])])
    vals = gap_widths[:, 0]
    widths = gap_widths[:, 1]
    starts = np.cumsum(widths) - widths

    # Compute the start frequency and width of the widest gap (in bins)
    open = (vals == 0)
    n_open = np.count_nonzero(open)
    n_used = np.count_nonzero(~open)
    n_fft = len(self.spectrogram[0])
    n_obs = 5
    open_starts = np.sort(starts[open])[::-1] / n_fft
    open_widths = np.sort(widths[open])[::-1] / n_fft
    used_starts = np.sort(starts[~open])[::-1] / n_fft
    used_widths = np.sort(widths[~open])[::-1] / n_fft

    open_start_obs = np.zeros(n_obs,)
    open_width_obs = np.zeros(n_obs,)
    used_start_obs = np.zeros(n_obs,)
    used_width_obs = np.zeros(n_obs,)
    open_start_obs[:min(n_open, n_obs)] = open_starts[:min(n_open, n_obs)]
    open_width_obs[:min(n_open, n_obs)] = open_widths[:min(n_open, n_obs)]
    used_start_obs[:min(n_used, n_obs)] = used_starts[:min(n_used, n_obs)]
    used_width_obs[:min(n_used, n_obs)] = used_widths[:min(n_used, n_obs)]

    obs = np.concatenate(
      (open_start_obs, open_width_obs, used_start_obs, used_width_obs))
    return obs.astype(np.float32)

  def _get_widest(self, spectrum: np.ndarray):
    # Group consecutive bins by value
    gap_widths = np.array([[x[0], len(list(x[1]))]
                          for x in itertools.groupby(spectrum)])
    vals = gap_widths[:, 0]
    widths = gap_widths[:, 1]
    starts = np.cumsum(widths) - widths

    # Compute the start frequency and width of the widest gap (in bins)
    open = (vals == 0)
    istart_widest = np.argmax(widths[open])
    widest_start = starts[open][istart_widest]
    widest_bw = widths[open][istart_widest]
    widest_stop = widest_start + widest_bw

    nfft = len(spectrum)
    widest_start = widest_start
    widest_stop = widest_stop
    return np.array([widest_start, widest_stop])

  def _render_frame(self) -> Optional[np.ndarray]:
    """
    Draw the current observation in a PyGame window if render_mode is 'human', or return the pixels as a numpy array if not.

    Returns
    -------
    Optional[np.ndarray]
        Grayscale pixel representation of the observation if render_mode is 'rgb_array', otherwise None.
    """
    if self.window is None and self.render_mode == "human":
      pygame.init()
      pygame.display.init()
      image_shape = (self.fft_size, self.n_image_snapshots)
      self.window = pygame.display.set_mode(image_shape)

    if self.clock is None and self.render_mode == "human":
      self.clock = pygame.time.Clock()

    # Draw canvas from pixels
    # The observation gets inverted here because I want black pixels on a white background.
    pixels = self.spectrogram.T.copy()
    pixels = pixels.reshape((-1, *pixels.shape[1:]))
    r = 96
    g = 28
    b = 3
    w = 255
    pixels[self.spectrogram.T == 1] = g
    pixels[self.radar_spectrogram.T == 1] = w
    overlap = np.logical_and(self.spectrogram.T == 1,
                             self.radar_spectrogram.T == 1)
    pixels[overlap] = r
    canvas = pygame.surfarray.make_surface(pixels)

    if self.render_mode == "human":
      # Copy canvas drawings to the window
      self.window.blit(canvas, canvas.get_rect())
      pygame.event.pump()

      pygame.display.update()

      # Ensure that human rendering occurs at the pre-defined framerate
      self.clock.tick(self.metadata["render_fps"])
    else:
      return pixels
