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
  metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}
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
    # Observation space has two channels. The first channel is the interference spectrogram and the second is the radar spectrogram
    self.observation_space = gym.spaces.Dict({
        'spectrogram': gym.spaces.Box(low=0, high=255, shape=(1, 50, 50), dtype=np.uint8),
        'spectrum': gym.spaces.Box(low=0, high=255, shape=(fft_size,), dtype=np.uint8)
    })

    # Action space is the start and span of the radar waveform
    self.action_space = gym.spaces.Box(
        low=0, high=1, shape=(2,), dtype=np.float32)
    
    self.filename = filename
    self.channel_bandwidth = channel_bandwidth
    self.fft_size = fft_size
    self.n_image_snapshots = n_image_snapshots
    self.render_mode = render_mode
    
    self.spectrogram_freq_axis = np.linspace(
        0, self.channel_bandwidth, self.observation_space['spectrogram'].shape[2])
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
    radar_start_freq = action[0] * self.channel_bandwidth
    radar_bw = np.clip(action[1] * self.channel_bandwidth,
                       0, self.channel_bandwidth - radar_start_freq)
    radar_stop_freq = radar_start_freq + radar_bw
    # Radar spectrum occupancy (with history)
    radar_spectrum = np.zeros(self.fft_size)
    radar_occupied = np.logical_and(self.fft_freq_axis >= radar_start_freq,
                                    self.fft_freq_axis <= radar_stop_freq)
    radar_spectrum[radar_occupied] = 1


    # Update the communications occupancy
    self.start_ind += 1
    stop_ind = self.start_ind + self.n_image_snapshots
    self.spectrogram = np.roll(self.spectrogram, -1, axis=0)
    self.spectrogram[-1] = self.data[self.start_ind]
    # "Ground truth" spectrum (without radar)
    self.current_spectrum = self.spectrogram[-1]
    # Add radar to the observation
    self.spectrogram[-1, radar_spectrum == 1] = 2

    # Compute reward
    bandwidth_reward = radar_bw / self.channel_bandwidth
    n_collision = np.count_nonzero(np.logical_and(self.current_spectrum, radar_spectrum))
    collision_penalty = -n_collision / self.fft_size
    reward = 0.4*bandwidth_reward + 0.6*collision_penalty

    # Propagate the environment forward a bit
    for _ in range(10):
      self.start_ind += 1
      stop_ind = self.start_ind + self.n_image_snapshots
      if stop_ind > self.data.shape[0]:
        break
      # To maintain a memory of where the radar transmitted, roll the spectrogram every time step
      self.spectrogram = np.roll(self.spectrogram, -1, axis=0)
      self.spectrogram[-1] = self.data[self.start_ind]
      self.current_spectrum = self.spectrogram[-1]

    # Downsample the image to use as an observation
    self.spectrogram_obs[0] = cv2.resize(
        self.spectrogram*255,
        self.observation_space['spectrogram'].shape[1:],
        interpolation=cv2.INTER_AREA)

    obs = {
        "spectrogram": self.spectrogram_obs,
        "spectrum": self.current_spectrum,
    }
    terminated = False
    # End the episode early if we reach the end of the data
    truncated = stop_ind >= self.data.shape[0]
    info = {
        'collision_penalty': collision_penalty,
        'occupancy_reward': bandwidth_reward
    }

    if self.render_mode == "human":
      self._render_frame()
    return obs, reward, terminated, truncated, info

  def reset(self, seed: int = None, options: dict = None):
    super().reset(seed=seed)

    # Randomly sample spectrogram from a file
    self.start_ind = self.np_random.integers(
        0, self.data.shape[0] - self.n_image_snapshots)
    stop_ind = self.start_ind + self.n_image_snapshots
    self.current_spectrum = self.data[self.start_ind]
    self.spectrogram = self.data[self.start_ind:stop_ind]
    self.spectrogram_obs = cv2.resize(
        self.spectrogram*255,
        self.observation_space['spectrogram'].shape[1:],
        interpolation=cv2.INTER_AREA)[np.newaxis, :]

    obs = {
        "spectrogram": self.spectrogram_obs,
        "spectrum": self.current_spectrum,
    }
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
    pixels[self.spectrogram.T == 2] = w
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
