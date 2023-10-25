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
  This is a modified version of the environment I created for the original paper submission. 
  
  # TODO: Merge discrete and continuous envs
  # TODO: Remove max bandwidth term in reward function?
  """
  metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

  def __init__(self, 
               render_mode = None, 
               nfft: int = 1024,
               pri: int = 10,
               collision_weight: float = 1,
               n_action_bins: int = None,
              #  max_collision: float = 0.005,
               n_pulse_cpi: int = 256,
               dataset: str = "/home/shane/data/hocae_snaps_2_4_cleaned_10_0.dat",
               order='C',
               seed: int = np.random.randint(0, 2**32 - 1)):
    # Parameters
    self.nfft = nfft
    self.pri = pri
    # self.max_collision = max_collision
    self.n_pulse_cpi = n_pulse_cpi
    self.dataset = dataset
    self.collision_weight = collision_weight
    
    self.np_random = np.random.RandomState(seed)

    self.interference = RecordedInterference(
        filename=dataset, order=order, fft_size=self.nfft, seed=seed)
    self.interference.data[:, nfft//2] = 0
    self.interference.data[:, 100:105] = 0
    self.interference.data[:, -105:-100] = 0
    self.freq_axis = np.linspace(0, 1, self.nfft)

    self.observation_space = gym.spaces.Box(low=0, high=1,
                                            shape=(self.pri, self.nfft,))
    if n_action_bins is not None:
      self.action_table = self._create_action_table(n_action_bins)
      self.action_space = gym.spaces.Discrete(self.action_table.shape[0])
      self.discrete_actions = True
    else:
      self.action_space = gym.spaces.Box(low=0, high=1, shape=(2,))
      self.discrete_actions = False
    
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
    self.missed_bandwidths = []
    self.cpi_bandwidths = []
    self.cpi_center_freqs = []
    self.bandwidth_diffs = []
    self.center_freq_diffs = []
    
    self.num_shift = self.np_random.randint(0, self.nfft)
    
    # Store last N observations
    self.history_len = 512
    self.history = {
        "radar": deque([np.zeros(self.nfft, dtype=np.uint8) for _ in range(self.history_len)], maxlen=self.history_len),
        "interference": deque([np.zeros(self.nfft, dtype=np.uint8) for _ in range(self.history_len)], maxlen=self.history_len),
    }

    self.interference.reset()
    obs = np.zeros((self.pri, self.nfft), dtype=np.float32)
    for i in range(self.pri):
      obs[i] = self.interference.step(self.time)
    obs = np.roll(obs, self.num_shift, axis=1)
    
    if self.render_mode == "human":
        for i in range(self.pri):
          self.history["interference"].append(obs[i])
          self.history["radar"].append(np.zeros(self.nfft, dtype=np.uint8))
        self._render_frame()
        
    return obs, {}

  def step(self, action: np.ndarray):
    if self.discrete_actions:
      action = self.action_table[action]
    # Compute radar spectrum
    start_freq = action[0]
    stop_freq = np.clip(action[1], start_freq, 1)
    radar_bw = stop_freq - start_freq
    fc = start_freq + radar_bw / 2
    radar_spectrum = np.logical_and(
        self.freq_axis > start_freq, self.freq_axis < stop_freq)
    
    obs = np.zeros((self.pri, self.nfft), dtype=np.float32)
    for i in range(self.pri):
      obs[i] = self.interference.step(self.time)
    obs = np.roll(obs, self.num_shift, axis=1)
    
    collision_bw = np.count_nonzero(
      np.logical_and(radar_spectrum == 1, obs[0] == 1)
    ) / self.nfft

    # Compute max bandwidth from contiguous zeros in obs[0]
    widest_start, widest_stop = self._get_widest(obs[0])
    widest_bw = (widest_stop - widest_start) / self.nfft
    reward = (radar_bw - widest_bw) - self.collision_weight*collision_bw 
    # if collision_bw <= self.max_collision else (0 - widest_bw) - self.collision_weight*collision_bw

    self.step_count += 1
    self.bandwidths.append(radar_bw)
    self.collisions.append(collision_bw)
    self.widests.append(widest_bw)
    self.missed_bandwidths.append(widest_bw - radar_bw)
    self.cpi_bandwidths.append(radar_bw)
    self.cpi_center_freqs.append(fc)
    self.bandwidth_diffs.append(abs(radar_bw - np.mean(self.cpi_bandwidths)))
    self.center_freq_diffs.append(abs(fc - np.mean(self.cpi_center_freqs)))

    info = {
        'mean_bw': np.mean(self.bandwidths),
        'mean_collision_bw': np.mean(self.collisions),
        'mean_widest_bw': np.mean(self.widests),
        'mean_missed_bw': np.mean(self.missed_bandwidths),
        'mean_bw_diff': np.mean(self.bandwidth_diffs),
        'mean_fc_diff': np.mean(self.center_freq_diffs),
        'pulse_index': (self.step_count % self.n_pulse_cpi) / self.n_pulse_cpi,
    }
    
    # Measure the bandwidth difference between pulses
    # TODO: Remove this and handle in reset()
    if self.step_count % self.n_pulse_cpi == 0:
      self.cpi_bandwidths.clear()
      self.cpi_center_freqs.clear()
      self.bandwidth_diffs.clear()
      self.center_freq_diffs.clear()
    
    if self.render_mode == "human":
      for i in range(self.pri):
        self.history["interference"].append(obs[i])
        if i == 0:
          self.history["radar"].append(radar_spectrum)
        else:
          self.history["radar"].append(np.zeros(self.nfft, dtype=np.uint8))
      self._render_frame()
        
    terminated = False
    truncated = False
    return obs, reward, terminated, truncated, info

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
    
  @staticmethod
  def _get_widest(spectrum: np.ndarray):
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
    return widest_start, widest_stop
  
  @staticmethod
  def _create_action_table(n_action_bins: int):
    n_actions = n_action_bins*(n_action_bins+1)//2
    action_table = np.zeros((n_actions, 2))
    action_index = 0
    for i in range(1, n_action_bins+1):
      x = np.zeros(n_action_bins)
      x[:i] = 1
      for j in range(n_action_bins-(i-1)):
        xr = np.roll(x, j)
        start = np.argmax(xr)
        stop = start + i
        action_table[action_index] = np.array([start, stop]) / n_action_bins
        action_index += 1
    return action_table

      
if __name__ == '__main__':
  # Computing SAA metrics. Don't need any kwargs
#   ENV_KWARGS = dict(
#     dataset="/home/shane/data/hocae_snaps_2_64ghz.dat",
#     pri=10,
#     order="F",
#     collision_weight=30,
#     # max_collision=0.005,
# )
  env = SpectrumEnv(dataset="/home/shane/data/hocae_snaps_2_64ghz.dat",
    pri=10,
    order="F",
    collision_weight=30,
  )

  obses = []
  obs, info = env.reset()
  
  metrics = dict(
    mean_bw=0,
    mean_collision_bw=0,
    mean_widest_bw=0,
    mean_missed_bw=0,
    mean_bw_diff=0,
    mean_fc_diff=0,
  )
  nsteps = 100_000
  for i in range(nsteps):
    start, stop = env._get_widest(obs[-1])
    action = np.array([start, stop]) / 1024
    obs, reward, term, trunc, info = env.step(action)
    for key in metrics.keys():
      metrics[key] += info[key]
      
    if i % 256 == 0:
      ob, info = env.reset()
      print(i)
      
  for key in metrics.keys():
    metrics[key] /= nsteps
    print(key, metrics[key])
    
