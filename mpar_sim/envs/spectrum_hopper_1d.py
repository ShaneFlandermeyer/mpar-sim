from typing import Optional
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pygame
import cv2

from mpar_sim.interference.single_tone import SingleToneInterferer


class SpectrumHopper1D(gym.Env):
  metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}
  """
  This gym environment formulates the interference avoidance problem as a continuous control task.
  
  Hopefully, the agent will be able to learn from "spectrograms" that have been pre-processed to form a binary mask indicating the presence of interference.
  """

  ##########################
  # Core Gym methods
  ##########################
  def __init__(self,
               render_mode: str = None):
    # TODO: Hard-coded for now
    self.channel_bw = 100e6
    self.interference = SingleToneInterferer(
        start_freq=0e6,
        bandwidth=20e6,
        duration=10e-3,
        duty_cycle=1,
    )

    # Observation space has two channels. The first channel is the interference spectrogram and the second is the radar spectrogram
    self.observation_space = gym.spaces.Box(
        low=0, high=1, shape=(1024,), dtype=np.uint8)

    # Action space is the start and span of the radar waveform
    self.action_space = gym.spaces.Box(
        low=0, high=1, shape=(2,), dtype=np.float32)

    # Note: Assuming time axis is discretized based on the PRF. This doesn't work if the PRF changes, but by that point I hope to be working on real-world data
    self.freq_axis = np.linspace(
        0, self.channel_bw, self.observation_space.shape[0])

    assert render_mode is None or render_mode in self.metadata["render_modes"]
    self.render_mode = render_mode
    # PyGame objects
    self.window = None
    self.clock = None

  def step(self, action: np.ndarray):
    # Update the radar waveform
    start_freq = action[0] * self.channel_bw
    radar_bw = min(action[1] * self.channel_bw, self.channel_bw - start_freq)

    # Update the communications interference
    n_freq_bins = np.digitize(
        self.interference.bandwidth, self.freq_axis - np.min(self.freq_axis))
    i_start_freq = np.digitize(
        self.interference.start_freq, self.freq_axis - np.min(self.freq_axis), right=True)
    i_stop_freq = i_start_freq + n_freq_bins
    self.spectrogram[i_start_freq:i_stop_freq] = 1

    occupancy_reward = radar_bw / self.channel_bw
    # For the collision reward, compute the overlap between the radar and comms spectrums
    interference_stop = self.interference.start_freq + self.interference.bandwidth
    radar_stop = start_freq + radar_bw
    collision_penalty = -(min(interference_stop, radar_stop) - \
          max(self.interference.start_freq, start_freq)) / self.channel_bw
    collision_penalty = min(collision_penalty, 0)
    reward = 1*occupancy_reward + 5*collision_penalty

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
