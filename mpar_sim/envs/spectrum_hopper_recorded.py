import copy
from typing import List, Optional
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pygame
import cv2
from mpar_sim.interference.interference import Interference
import itertools

from mpar_sim.interference.single_tone import SingleToneInterference


class SpectrumHopperRecorded(gym.Env):
  metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}
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
               render_mode: str = None) -> None:
    self.observation_space = gym.spaces.Box(
        low=0, high=255, shape=(frame_stack, n_image_snapshots, fft_size), dtype=np.uint8)

    # Action space is the start and span of the radar waveform
    self.action_space = gym.spaces.Box(
        low=0, high=1, shape=(2,), dtype=np.float32)

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
    self.data = self.data.reshape((n_snapshots, fft_size), order='F')
    # Zero out the dc component and X310 spur. Messes up the visualization and bandwidth computations
    idc = int(self.fft_size/2)
    ispur = np.digitize(10e6, self.fft_freq_axis)
    self.data[:, idc] = 0
    self.data[:, ispur-1:ispur+1] = 0

    assert render_mode is None or render_mode in self.metadata["render_modes"]
    self.render_mode = render_mode
    # PyGame objects
    self.window = None
    self.clock = None

  def step(self, action: np.ndarray):
    # Update the radar waveform
    # unsquashed_action = self.action_scale*action + self.action_bias
    radar_start_freq = action[0] * self.channel_bandwidth
    radar_bw = np.clip(action[1] * self.channel_bandwidth,
                       0, self.channel_bandwidth - radar_start_freq)
    radar_stop_freq = radar_start_freq + radar_bw

    # Radar spectrum occupancy (with history)
    radar_spectrum = np.zeros(self.fft_size)
    radar_occupied = np.logical_and(self.fft_freq_axis >= radar_start_freq,
                                    self.fft_freq_axis <= radar_stop_freq)
    radar_spectrum[radar_occupied] = 1
    self.radar_spectrogram = np.roll(self.radar_spectrogram, -1, axis=0)
    self.radar_spectrogram[-1] = radar_spectrum

    # Update the communications occupancy
    self.start_ind = (self.start_ind + 1) % self.data.shape[0]
    self.current_spectrum, widest_gap = self._merge_bins(self.data[self.start_ind])
    # Set spectrogram values
    self.spectrogram = np.roll(self.spectrogram, -1, axis=0)
    self.spectrogram[-1] = self.current_spectrum

    # Penalize the radar for collisions and missed opportunities
    n_radar_bins = np.count_nonzero(radar_spectrum)
    n_int_bins = np.count_nonzero(self.current_spectrum)
    intersection = np.count_nonzero(
        np.logical_and(self.current_spectrum, radar_spectrum))
    collision_penalty = intersection / n_int_bins if n_int_bins > 0 else 0
    missed_penalty = (widest_gap - n_radar_bins) / self.fft_size
    # Limit the penalties to the same scale
    collision_penalty = np.clip(collision_penalty, 0, 1)
    missed_penalty = np.clip(missed_penalty, 0, 1)
    # TODO: Trying to make reward positive
    reward = -(0.5*collision_penalty + 0.5*missed_penalty)

    # Propagate the environment forward a bit
    for _ in range(10):
      self.start_ind = (self.start_ind + 1) % self.data.shape[0]
      self.current_spectrum = self.data[self.start_ind]
      self.spectrogram = np.roll(self.spectrogram, -1, axis=0)
      self.spectrogram[-1] = self.current_spectrum
      self.spectrogram[-1], _ = self._merge_bins(self.spectrogram[-1])
      self.radar_spectrogram = np.roll(self.radar_spectrogram, -1, axis=0)
      self.radar_spectrogram[-1] = 0

    obs = self.spectrogram[np.newaxis, :]
    terminated = False
    # End the episode early if we reach the end of the data
    truncated = False
    info = {}

    if self.render_mode == "human":
      self._render_frame()
    return obs, reward, terminated, truncated, info

  def reset(self, seed: int = None, options: dict = None):
    super().reset(seed=seed)

    # Flip the data to get more diversity in the training
    axis = self.np_random.choice([0, 1])
    self.data = np.flip(self.data, axis=axis)

    # Randomly sample spectrogram from a file
    self.start_ind = self.np_random.integers(
        0, self.data.shape[0] - self.n_image_snapshots)
    stop_ind = self.start_ind + self.n_image_snapshots
    self.spectrogram = self.data[self.start_ind:stop_ind]
    self.radar_spectrogram = np.zeros_like(self.spectrogram)

    obs = self.spectrogram[np.newaxis, :]
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
    r = 96
    g = 28
    b = 3
    w = 255
    pixels[self.spectrogram.T == 1] = g
    pixels[self.radar_spectrogram.T == 1] = w
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

  def _merge_bins(self, spectrum, threshold=10):
    # Merge bins if they are separated by less than 5 zeros
    gap_widths = [(x[0], len(list(x[1])))
                  for x in itertools.groupby(spectrum)]
    istart = 0
    for val, width in gap_widths:
      if val == 0 and width < threshold:
        spectrum[istart:istart+width] = 1
      istart += width
    widest_gap = max(gap_widths, key=lambda x: x[1])[1]
    return spectrum, widest_gap
  
if __name__ == '__main__':
  env = SpectrumHopperRecorded(
      filename='/home/shane/data/HOCAE_Snaps_bool.dat',
      channel_bandwidth=100e6,
      fft_size=1024,
      n_image_snapshots=500,
      render_mode='human')
  obs, info = env.reset()
  x = 1
  plt.imshow(obs[0])
  plt.colorbar()
  plt.show()
