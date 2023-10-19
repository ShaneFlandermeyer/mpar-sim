from abc import ABCMeta
import itertools
import gymnasium as gym
from matplotlib import pyplot as plt
import numpy as np
from mpar_sim.interference.recorded import RecordedInterference
from mpar_sim.interference.single_tone import SingleToneInterference

class SpectrumEnv(gym.Env):
  """
  This is a modified version of the environment I created for the original paper submission. Here, 
  """
  metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

  def __init__(self, render_mode=None, seed: int = None):
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

  def reset(self, seed: int = None, options=None):
    # Counters
    self.step_count = 0
    self.time = 0
    # Metrics
    self.mean_bw = 0
    self.mean_collision_bw = 0
    self.mean_widest_bw = 0
    self.mean_bw_diff = 0

    self.interference.reset()
    obs = np.zeros((self.pri, self.fft_size), dtype=np.float32)
    for i in range(self.pri):
      obs[i] = self.interference.step(self.time)
      
    return obs, {}

  def step(self, action: np.ndarray):
    obs = np.zeros((self.pri, self.fft_size), dtype=np.float32)
    for i in range(self.pri):
      obs[i] = self.interference.step(self.time)
      
    # TODO: Compute reward
    start_freq = action[0]
    stop_freq = np.clip(action[1], start_freq, 1)
    bandwidth = stop_freq - start_freq
    fc = start_freq + bandwidth / 2
    spectrum = np.logical_and(
        self.freq_axis >= start_freq, self.freq_axis <= stop_freq)
    collision_bw = np.count_nonzero(
      np.logical_and(spectrum == 1, obs[0] == 1)
    ) / self.fft_size

    # Compute max bandwidth from contiguous zeros in obs[0]
    widest = self._get_widest(obs[0])
    widest_bw = (widest[1] - widest[0]) / self.fft_size
    # TODO: Test the subtraction of the collision BW before pushing
    reward = (bandwidth - widest_bw - collision_bw) if collision_bw <= self.max_collision else -widest_bw

    self.step_count += 1
    self.mean_bw = (self.mean_bw * (self.step_count - 1) + bandwidth) / self.step_count
    self.mean_collision_bw = (self.mean_collision_bw * (self.step_count - 1) + collision_bw) / self.step_count
    self.mean_widest_bw = (self.mean_widest_bw * (self.step_count - 1) + widest_bw) / self.step_count
    self.mean_bw_diff = (self.mean_bw_diff * (self.step_count - 1) + (bandwidth - widest_bw)) / self.step_count

    terminated = False
    truncated = False
    info = {
        'mean_bw': self.mean_bw,
        'mean_collision_bw': self.mean_collision_bw,
        'mean_widest_bw': self.mean_widest_bw,
        'mean_bw_diff': self.mean_bw_diff,
    }
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


if __name__ == '__main__':
  env = gym.vector.SyncVectorEnv([lambda: SpectrumEnv()])

  obs, info = env.reset()
  obs, reward, term, trunc, info = env.step(np.array([[0, 1]]))

  plt.imshow(obs[0])
  plt.savefig('./test.png')
  # plt.show()
