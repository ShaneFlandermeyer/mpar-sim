from abc import ABCMeta
import itertools
import cv2
import gymnasium as gym
from matplotlib import pyplot as plt
import numpy as np
import pygame
from mpar_sim.interference.recorded import RecordedInterference
from mpar_sim.interference.single_tone import SingleToneInterference
from collections import deque

class SpectrumEnv(gym.Env):
  """
  This is a modified version of the environment I created for the original paper submission. Here, 
  """
  metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

  def __init__(self, render_mode = None, seed: int = None):
    # Parameters
    self.fft_size = 1024
    self.pri = 10
    self.max_collision = 0.01
    self.pulse_per_cpi = 256
    # self.min_bw = 0.1

    self.interference = RecordedInterference(
        "/home/shane/data/hocae_snaps_2_4_cleaned_10_0.dat", self.fft_size, seed=seed)
    self.freq_axis = np.linspace(0, 1, self.fft_size)

    self.observation_space = gym.spaces.Box(low=0, high=1,
                                            shape=(self.pri, self.fft_size,))
    self.action_space = gym.spaces.Box(low=0, high=1, shape=(2,))
    
    assert render_mode is None or render_mode in self.metadata["render_modes"]
    self.render_mode = render_mode
    # Pygame setup
    self.window_size = (512, 512)
    self.window = None
    self.clock = None

  def reset(self, seed: int = None, options=None):
    # Counters
    self.step_count = 0
    self.time = 0
    # Metrics
    self.bandwidths = []
    self.collisions = []
    self.widests = []
    self.bw_diffs = []
    # TODO: Applying random shift
    self.num_shift = np.random.randint(0, self.fft_size)
    
    # Store last N observations
    self.history_len = 512
    self.history = {
        "radar": deque([np.zeros(self.fft_size, dtype=np.uint8) for _ in range(self.history_len)], maxlen=self.history_len),
        "interference": deque([np.zeros(self.fft_size, dtype=np.uint8) for _ in range(self.history_len)], maxlen=self.history_len),
    }

    self.interference.reset()
    obs = np.zeros((self.pri, self.fft_size), dtype=np.float32)
    for i in range(self.pri):
      obs[i] = self.interference.step(self.time)
    obs = np.roll(obs, self.num_shift, axis=1)
    
    if self.render_mode == "human":
        for i in range(self.pri):
          self.history["interference"].append(obs[i])
          self.history["radar"].append(np.zeros(self.fft_size, dtype=np.uint8))
        self._render_frame()
        
    return obs, {}

  def step(self, action: np.ndarray):
    # TODO: Compute reward
    start_freq = action[0]
    stop_freq = np.clip(action[1], start_freq, 1)
    bandwidth = stop_freq - start_freq
    fc = start_freq + bandwidth / 2
    spectrum = np.logical_and(
        self.freq_axis > start_freq, self.freq_axis < stop_freq)
    obs = np.zeros((self.pri, self.fft_size), dtype=np.float32)
    for i in range(self.pri):
      obs[i] = self.interference.step(self.time)
    obs = np.roll(obs, self.num_shift, axis=1)
    
    collision_bw = np.count_nonzero(
      np.logical_and(spectrum == 1, obs[0] == 1)
    ) / self.fft_size

    # Compute max bandwidth from contiguous zeros in obs[0]
    widest = self._get_widest(obs[0])
    widest_bw = (widest[1] - widest[0]) / self.fft_size
    reward = (bandwidth - widest_bw - collision_bw) if collision_bw <= self.max_collision else (0 - widest_bw - collision_bw)

    self.step_count += 1
    self.bandwidths.append(bandwidth)
    self.collisions.append(collision_bw)
    self.widests.append(widest_bw)
    self.bw_diffs.append(bandwidth - widest_bw)

    terminated = False
    truncated = False
    info = {
        'mean_bw': np.mean(self.bandwidths),
        'mean_collision_bw': np.mean(self.collisions),
        'mean_widest_bw': np.mean(self.widests),
        'mean_bw_diff': np.mean(self.bw_diffs),
    }
    
    if self.render_mode == "human":
      for i in range(self.pri):
        self.history["interference"].append(obs[i])
        if i == 0:
          self.history["radar"].append(spectrum)
        else:
          self.history["radar"].append(np.zeros(self.fft_size, dtype=np.uint8))
      self._render_frame()
        
    return obs, reward, terminated, truncated, info
  
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

  def render(self):
    if self.render_mode == "rgb_array":
      return self._render_frame()

  def close(self):
    if self.render_mode == "human":
      pygame.quit()
      
  def _render_frame(self):
    if self.window is None and self.render_mode == "human":
      pygame.init()
      pygame.display.init()
      self.window = pygame.display.set_mode(self.window_size)

    if self.clock is None and self.render_mode == "human":
      self.clock = pygame.time.Clock()

    radar_spectrogram = np.stack(self.history["radar"], axis=0)
    interf_spectrogram = np.stack(self.history["interference"], axis=0)
    intersection = np.logical_and(radar_spectrogram.T, interf_spectrogram.T)

    pixels = interf_spectrogram.T*100
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

      
if __name__ == '__main__':
  env = gym.vector.SyncVectorEnv([lambda: SpectrumEnv()])

  obs, info = env.reset()
  obs, reward, term, trunc, info = env.step(np.array([[0, 1]]))

  plt.imshow(obs[0])
  plt.savefig('./test.png')
  # plt.show()
