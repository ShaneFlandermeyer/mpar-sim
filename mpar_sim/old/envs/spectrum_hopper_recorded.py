import copy
from typing import List, Optional
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pygame
import cv2
import itertools


class SpectrumHopperRecorded(gym.Env):
  metadata = {"render_modes": ["human", "rgb_array"], "obs_modes": [
      "spectrogram", "fft"], "render_fps": 50}
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
               obs_mode: str = "spectrogram",
               image_shape=(64, 64),
               render_mode: str = None) -> None:
    assert obs_mode in self.metadata["obs_modes"]
    if obs_mode == "spectrogram":
      self.observation_space = gym.spaces.Box(
          low=0, high=255, shape=(2, *image_shape), dtype=np.uint8)
    else:
      self.observation_space = gym.spaces.Box(
          low=0, high=1, shape=(fft_size, ), dtype=np.uint8)
    self.obs_mode = obs_mode

    # Action space is the start and span of the radar waveform
    self.action_space = gym.spaces.Box(
        low=np.array([-1, -1]), high=np.array([1, 1]), shape=(2,), dtype=np.float32)

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

    assert render_mode is None or render_mode in self.metadata["render_modes"]
    self.render_mode = render_mode
    # PyGame objects
    self.window = None
    self.clock = None

  def step(self, action: np.ndarray):
    # Update the radar waveform
    radar_start_freq = np.clip(action[0] * self.channel_bandwidth/2,
                               None, self.channel_bandwidth/2)
    radar_stop_freq = np.clip(action[1] * self.channel_bandwidth/2,
                              radar_start_freq, None)

    # Radar spectrum occupancy (with history)
    self.radar_spectrogram = np.roll(self.radar_spectrogram, -1, axis=0)
    self.radar_spectrogram[-1] = np.logical_and(
        self.fft_freq_axis >= radar_start_freq,
        self.fft_freq_axis <= radar_stop_freq)

    # Update the communications occupancy
    self.spectrogram = np.roll(self.spectrogram, -1, axis=0)
    self.start_ind = (self.start_ind + 1) % self.data.shape[0]
    self.spectrogram[-1] = self.data[self.start_ind]
    # TODO: In final env, reward should be computed AFTER the spectrogram is updated
    reward = self._compute_reward(
        self.radar_spectrogram[-1], self.spectrogram[-1])

    # "Scroll" the spectrogram to the next PRI
    n_roll = 20
    for _ in range(n_roll):
      self.spectrogram = np.roll(self.spectrogram, -n_roll, axis=0)
      self.start_ind = (self.start_ind + 1) % self.data.shape[0]
      self.spectrogram[-1] = self.data[self.start_ind]
      self.radar_spectrogram = np.roll(self.radar_spectrogram, -1, axis=0)
      self.radar_spectrogram[-1] = 0

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
    total_length = self.n_image_snapshots
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
      spectro = cv2.resize(255*self.spectrogram, (64, 64), interpolation=cv2.INTER_AREA).reshape((1, 64, 64))
      radar = cv2.resize(255*self.radar_spectrogram, (64, 64), interpolation=cv2.INTER_AREA).reshape((1, 64, 64))
      obs = np.concatenate((spectro, radar), axis=0)
    else:
      obs = self.spectrogram[-1]
    return obs.astype(np.uint8)

  def _compute_reward(self,
                      radar_spectrum: np.ndarray,
                      interference: np.ndarray,
                      ):
    _, n_widest = self._get_widest(interference)
    n_radar_bins = np.sum(radar_spectrum)
    n_collisions = np.sum(np.logical_and(radar_spectrum, interference))
    reward = (n_radar_bins - n_widest) / self.fft_size if n_collisions < 10 else -n_widest/self.fft_size
    return reward

  def _get_widest(self, spectrum):
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


if __name__ == '__main__':
  env = SpectrumHopperRecorded(
      filename='/home/shane/data/HOCAE_Snaps_bool_cleaned.dat',
      channel_bandwidth=100e6,
      fft_size=1024,
      n_image_snapshots=256,
      obs_mode='spectrogram',
      render_mode='human',)
  env = gym.wrappers.TimeLimit(env, max_episode_steps=250)
  # env = gym.wrappers.ResizeObservation(env, (64, 64))
  # env = gym.wrappers.TransformObservation(env, lambda x: x.squeeze(-1))
  # env = gym.wrappers.FrameStack(env, 3)

  obs, info = env.reset()
  for i in range(10):
    obs, r, t, t2, i = env.step(np.array([-0.5, 0.5]))
    # obs, reward, term, trunc, = env.step(env.action_space.sample())
  # x = 1
  plt.figure()
  plt.imshow(obs[0], aspect='auto')
  plt.colorbar()
  # plt.savefig('./test.png')
#
  plt.figure()
  plt.imshow(obs[1])
  plt.colorbar()
  plt.show()
