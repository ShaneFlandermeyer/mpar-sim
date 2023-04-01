import copy
from typing import List, Optional
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pygame
import cv2
import itertools


class SpectrumHopperRecorded(gym.Env):
  metadata = {"render_modes": ["human", "rgb_array"], "obs_modes": ["spectrogram", "fft"], "render_fps": 50}
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
               frame_stack: int = 1,
               obs_mode: str = "spectrogram",
               render_mode: str = None) -> None:
    assert obs_mode in self.metadata["obs_modes"]
    if obs_mode == "spectrogram":
      self.observation_space = gym.spaces.Box(
          low=0, high=255, shape=(n_image_snapshots, fft_size, frame_stack), dtype=np.uint8)
    else:
      self.observation_space = gym.spaces.Box(
          low=0, high=1, shape=(fft_size, ), dtype=np.uint8)
    self.obs_mode = obs_mode

    # Action space is the start and span of the radar waveform
    self.action_space = gym.spaces.Box(
        low=np.array([0, 0]), high=np.array([1, 1]), shape=(2,), dtype=np.float32)

    self.filename = filename
    self.channel_bandwidth = channel_bandwidth
    self.fft_size = fft_size
    self.n_image_snapshots = n_image_snapshots
    self.frame_stack = frame_stack
    self.render_mode = render_mode

    self.fft_freq_axis = np.linspace(
        0, self.channel_bandwidth, self.fft_size)

    self.data = np.fromfile(filename, dtype=np.uint8)
    n_snapshots = int(self.data.size / fft_size)
    self.data = self.data.reshape((n_snapshots, fft_size), order='C')

    assert render_mode is None or render_mode in self.metadata["render_modes"]
    self.render_mode = render_mode
    # PyGame objects
    self.window = None
    self.clock = None

  def step(self, action: np.ndarray):
    # Update the radar waveform
    radar_start_freq = action[0] * self.channel_bandwidth
    radar_bw = np.clip(action[1] * self.channel_bandwidth,
                       0, self.channel_bandwidth - radar_start_freq)
    radar_stop_freq = radar_start_freq + radar_bw
    # Radar spectrum occupancy (with history)
    self.radar_spectrogram = np.roll(self.radar_spectrogram, -1, axis=0)
    self.radar_spectrogram[-1] = np.logical_and(
        self.fft_freq_axis >= radar_start_freq,
        self.fft_freq_axis <= radar_stop_freq)

    # Update the communications occupancy
    self.spectrogram = np.roll(self.spectrogram, -1, axis=0)
    self.start_ind = (self.start_ind + 1) % self.data.shape[0]
    self.spectrogram[-1] = self.data[self.start_ind]
    _, widest_bw_bins = self._get_widest(self.spectrogram[-1])
    reward = self._compute_reward(
        self.radar_spectrogram[-1], self.spectrogram[-1], widest_bw_bins)

    # Propagate the environment forward a bit
    # for _ in range(10):
    #   self.start_ind = (self.start_ind + 1) % self.data.shape[0]
    #   self.spectrogram = np.roll(self.spectrogram, -1, axis=0)
    #   self.spectrogram[-1] = self.data[self.start_ind]

    #   self.radar_spectrogram = np.roll(self.radar_spectrogram, -1, axis=0)
    #   self.radar_spectrogram[-1] = 0

    # Re-order into frames, then swap the axes to get the correct output shape
    obs = self._get_obs()
    info = {}
    terminated = False
    truncated = False

    if self.render_mode == "human":
      self._render_frame()
    return obs, reward, terminated, truncated, info

  def reset(self, seed: int = None, options: dict = None):
    super().reset(seed=seed)

    # Shift the data to get more diversity in the training
    self.data = np.roll(self.data, self.np_random.integers(0, 512), axis=1)

    # Randomly sample spectrogram from a file
    total_length = self.n_image_snapshots * self.frame_stack
    self.start_ind = self.np_random.integers(
        0, self.data.shape[0] - total_length)
    stop_ind = self.start_ind + total_length
    self.spectrogram = self.data[self.start_ind:stop_ind]
    self.radar_spectrogram = np.zeros_like(self.spectrogram)

    # Re-order into frames, then swap the axes to get the correct output shape
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

  def _get_obs(self):
    if self.obs_mode == "spectrogram":
      obs = 255*self.spectrogram.reshape(
          (self.frame_stack, self.n_image_snapshots, self.fft_size))
      obs = np.transpose(obs, (1, 2, 0))
    else:
      obs = self.spectrogram[-1]
    return obs

  def _compute_reward(self,
                      radar_spectrum: np.ndarray,
                      interference: np.ndarray,
                      widest_bw_bins):
    # TODO: Only penalize collisions the agent can "see"
    collisions = np.logical_and(radar_spectrum, interference)
    n_radar_bins = np.sum(radar_spectrum)
    n_collisions = np.sum(collisions)
    return (n_radar_bins - 5*n_collisions) / self.fft_size
    # return reward
    # Reward the agent for starting in a location with a lot of open bandwidth and for utilizing the bandwidth without collision
    radar_nonzero = np.flatnonzero(radar_spectrum)
    if len(radar_nonzero) == 0:
      reward = 0
      info = {
          'start_reward': 0,
          'bw_reward': 0,
      }
      return reward, info
    istart, istop = radar_nonzero[0], radar_nonzero[-1]
    radar_bw_bins = istop - istart
    open_bin_inds = np.flatnonzero(self.current_spectrum[istart:])
    n_open_bins = open_bin_inds[0] if len(open_bin_inds) > 0 else 0
    start_reward = n_open_bins / widest_bw_bins
    # Penalize the bandwidth selection if it exceeds the available bandwidth
    if n_open_bins == 0:
      bw_reward = 0
    else:
      bw_reward = max(radar_bw_bins, n_open_bins) / n_open_bins + \
          min(n_open_bins - radar_bw_bins, 0) / n_open_bins

    reward = start_reward + bw_reward
    info = {
        'start_reward': start_reward,
        'bw_reward': bw_reward,
    }
    return reward, info

  def _get_widest(self, spectrum):
    # Group consecutive bins by value
    gap_widths = np.array([[x[0], len(list(x[1]))]
                          for x in itertools.groupby(spectrum)])
    widths = gap_widths[:, 1]
    starts = np.cumsum(widths) - widths

    # Compute the start frequency and width of the widest gap (in bins)
    istart_widest = np.argmax(widths)
    widest_start = starts[istart_widest]
    widest_bw = widths[istart_widest]

    return widest_start, widest_bw

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
      image_shape = (self.fft_size, self.n_image_snapshots*self.frame_stack)
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


if __name__ == '__main__':
  env = SpectrumHopperRecorded(
      filename='/home/shane/data/HOCAE_Snaps_bool_cleaned.dat',
      channel_bandwidth=100e6,
      fft_size=1024,
      n_image_snapshots=200,
      frame_stack=3,
      render_mode='human')
  obs, info = env.reset()
  # obs, reward, term, trunc, = env.step(env.action_space.sample())
  # x = 1
  plt.figure()
  plt.imshow(obs[:, :, 0])
  plt.colorbar()
  plt.figure()
  plt.imshow(obs[:, :, 1])
  plt.colorbar()
  plt.show()
