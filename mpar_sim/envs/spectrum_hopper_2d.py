from typing import List, Optional
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pygame
import cv2
from mpar_sim.interference.interference import Interference

from mpar_sim.interference.single_tone import SingleToneInterference


class SpectrumHopper2D(gym.Env):
  metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}
  """
  This gym environment formulates the interference avoidance problem as a continuous control task.
  
  Hopefully, the agent will be able to learn from "spectrograms" that have been pre-processed to form a binary mask indicating the presence of interference.
  """

  ##########################
  # Core Gym methods
  ##########################
  def __init__(self,
               interference: List[Interference],
               channel_bandwidth: float = 100e6,
               fft_size: int = 1024,
               render_mode: str = None) -> None:
    self.interference = interference
    if not isinstance(interference, list):
      self.interference = [interference]
    self.channel_bandwidth = channel_bandwidth
    self.fft_size = fft_size
    self.render_mode = render_mode

    # Observation space has two channels. The first channel is the interference spectrogram and the second is the radar spectrogram
    self.observation_space = gym.spaces.Box(
        low=0, high=255, shape=(1, 84, 84), dtype=np.uint8)

    # Action space is the start and span of the radar waveform
    self.action_space = gym.spaces.Box(
        low=0, high=1, shape=(2,), dtype=np.float32)

    # # Note: Assuming time axis is discretized based on the PRF. This doesn't work if the PRF changes, but by that point I hope to be working on real-world data
    self.spectrogram_freq_axis = np.linspace(
        0, self.channel_bandwidth, self.observation_space.shape[2])
    self.fft_freq_axis = np.linspace(
        0, self.channel_bandwidth, self.fft_size)

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

    # Update the communications occupancy
    self.spectrogram[0] = np.roll(self.spectrogram[0], -1, axis=0)
    self.spectrogram[0, -1, :] = 0
    self.current_spectrum[:] = 0
    self._update_spectrum()

    # Radar spectrum occupancy
    self.radar_spectrogram[0] = np.roll(self.radar_spectrogram[0], -1, axis=0)
    self.radar_spectrogram[0, -1, :] = 0
    radar_occupied = np.logical_and(
        self.spectrogram_freq_axis >= radar_start_freq,
        self.spectrogram_freq_axis <= radar_stop_freq)
    self.radar_spectrogram[0, -1, radar_occupied] = 1
    
    radar_occupied = np.logical_and(self.fft_freq_axis >= radar_start_freq,
                                    self.fft_freq_axis <= radar_stop_freq)
    self.radar_spectrum[:] = 0
    self.radar_spectrum[radar_occupied] = 1

    # Compute reward
    occupancy_reward = radar_bw / self.channel_bandwidth
    n_occupied = np.count_nonzero(self.current_spectrum)
    collision_penalty = -np.sum(
        np.logical_and(self.current_spectrum, self.radar_spectrum)) / n_occupied
    collision_penalty = np.clip(collision_penalty, -np.Inf, 0)
    reward = 1*occupancy_reward + 1*collision_penalty

    self.time += 1

    obs = self.spectrogram
    terminated = False
    truncated = False
    info = {
        'collision_penalty': collision_penalty,
        'occupancy_reward': occupancy_reward
    }

    if self.render_mode == "human":
      self._render_frame()
    return obs, reward, terminated, truncated, info

  def reset(self, seed: int = None, options: dict = None):
    super().reset(seed=seed)

    for interferer in self.interference:
      interferer.reset()

    self.spectrogram = np.zeros(self.observation_space.shape, dtype=np.uint8)
    self.current_spectrum = np.zeros(self.fft_size, dtype=np.uint8)
    self.radar_spectrogram = np.zeros(
        self.observation_space.shape, dtype=np.uint8)
    self.radar_spectrum = np.zeros(self.fft_size, dtype=np.uint8)
    self.time = 0
    # Run a sequence of no-ops to get a full spectrogram
    for _ in range(self.observation_space.shape[1]):
      self.spectrogram[0] = np.roll(self.spectrogram[0], -1, axis=0)
      self.spectrogram[0, -1, :] = 0
      self._update_spectrum()
      self.time += 1

    observation = self.spectrogram
    info = {}

    if self.render_mode == "human":
      self._render_frame()

    return observation, info

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
  def _update_spectrum(self) -> None:
    for interferer in self.interference:
      interferer.step(time=self.time)
      if interferer.is_active:
        interference_stop = interferer.start_freq + interferer.bandwidth
        spectro_occupied = np.logical_and(
            self.spectrogram_freq_axis >= interferer.start_freq,
            self.spectrogram_freq_axis <= interference_stop)
        self.spectrogram[0, -1, spectro_occupied] = 1
        # Also check occupation for the reward computation
        fft_occupied = np.logical_and(
            self.fft_freq_axis >= interferer.start_freq,
            self.fft_freq_axis <= interference_stop)
        self.current_spectrum[fft_occupied] = 1

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
      self.window = pygame.display.set_mode((256, 256))

    if self.clock is None and self.render_mode == "human":
      self.clock = pygame.time.Clock()

    # Draw canvas from pixels
    # The observation gets inverted here because I want black pixels on a white background.
    pixels = self.spectrogram[0].T*100
    pixels[self.radar_spectrogram[0].T == 1] = 255
    pixels = cv2.resize(pixels, (256, 256), interpolation=cv2.INTER_NEAREST)
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
