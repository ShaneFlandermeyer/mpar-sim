from typing import List, Optional
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pygame
import cv2
from mpar_sim.interference.interference import Interference

from mpar_sim.interference.single_tone import SingleToneInterference


class SpectrumHopper1D(gym.Env):
  metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}
  """
  This gym environment formulates the interference avoidance problem as a continuous control task.
  
  Hopefully, the agent will be able to learn from "spectrograms" that have been pre-processed to form a binary mask indicating the presence of interference.
  
  TODO: For the 1D case, need some sort of recurrence to resolve multi-step interference
  """

  ##########################
  # Core Gym methods
  ##########################
  def __init__(self,
               interference: List[Interference],
               channel_bandwidth: float = 100e6,
               render_mode: str = None):
    self.channel_bandwidth = channel_bandwidth

    self.interference = interference
    if not isinstance(interference, list):
      self.interference = [interference]

    # Observation space has two channels. The first channel is the interference spectrogram and the second is the radar spectrogram
    self.observation_space = gym.spaces.Box(
        low=0, high=1, shape=(100,), dtype=np.uint8)

    # Action space is the start and span of the radar waveform
    self.action_space = gym.spaces.Box(
        low=0, high=1, shape=(2,), dtype=np.float32)

    # Note: Assuming time axis is discretized based on the PRF. This doesn't work if the PRF changes, but by that point I hope to be working on real-world data
    self.freq_axis = np.linspace(
        0, self.channel_bandwidth, self.observation_space.shape[0])

    assert render_mode is None or render_mode in self.metadata["render_modes"]
    self.render_mode = render_mode
    # PyGame objects
    self.window = None
    self.clock = None

  def step(self, action: np.ndarray):
    # Update the radar waveform
    start_freq = action[0] * self.channel_bandwidth
    radar_bw = min(action[1] * self.channel_bandwidth,
                   self.channel_bandwidth - start_freq)
    radar_stop = start_freq + radar_bw
    # Radar spectrum occupancy
    occupied = np.logical_and(self.freq_axis >= start_freq,
                              self.freq_axis <= radar_stop)
    radar_spectrum = np.zeros(self.observation_space.shape[0])
    radar_spectrum[occupied] = 1

    # Update the communications occupancy
    for interferer in self.interference:
      interferer.step(time=self.time)
      if interferer.is_active:
        interference_stop = interferer.start_freq + interferer.bandwidth
        occupied = np.logical_and(self.freq_axis >= interferer.start_freq,
                                  self.freq_axis <= interference_stop)
        self.spectrogram[occupied] = 1

    # Compute reward
    occupancy_reward = radar_bw / self.channel_bandwidth
    collision_penalty = -np.sum(np.logical_and(self.spectrogram == 1,
                                               radar_spectrum == 1)) / self.observation_space.shape[0]
    collision_penalty = min(collision_penalty, 0)
    reward = 1*occupancy_reward + 2*collision_penalty

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
    # Reset the spectrogram and time counter
    self.spectrogram = np.zeros(self.observation_space.shape, dtype=np.uint8)
    self.time = 0

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
  # def _

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
    pixels = self.spectrogram[0].T
    pixels[pixels == 255] = 100
    pixels[self.spectrogram[1].T == 255] = 255
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


if __name__ == '__main__':
  env = SpectrumHopper()
  obs, info = env.reset()
  print(env.action_space)
  print(env.observation_space)

  for i in range(4):
    start_freq = np.random.uniform(0, 1)
    bandwidth = np.random.uniform(0, 1)
    obs, reward, term, trunc, info = env.step([start_freq, bandwidth])

  plt.figure()
  plt.imshow(obs[1],
             extent=(env.freq_axis[0],
             env.freq_axis[-1], env.time_axis[0]*1e3, env.time_axis[-1]*1e3),
             aspect='auto')
  plt.colorbar()
  plt.show()
