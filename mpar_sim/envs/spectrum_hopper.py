import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

from mpar_sim.interference.single_tone import SingleToneInterferer


class SpectrumHopper(gym.Env):
  metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}
  """
  This gym environment formulates the interference avoidance problem as a continuous control task.
  
  Hopefully, the agent will be able to learn from "spectrograms" that have been pre-processed to form a binary mask indicating the presence of interference.
  """

  ##########################
  # Core Gym methods
  ##########################
  def __init__(self):
    # TODO: Hard-coded for now
    self.channel_bw = 100e6
    self.interference = SingleToneInterferer(
        start_freq=40e6,
        bandwidth=20e6,
        duration=10e-3,
        duty_cycle=0.5,
    )
    self.spectrogram_duration = 30e-3
    self.pri = 409.6e-6
    self.pulsewidth = 10e-6

    # Observation space has two channels. The first channel is the interference spectrogram and the second is the radar spectrogram
    self.observation_space = gym.spaces.Box(
        low=0, high=255, shape=(84, 84, 2), dtype=np.uint8)

    # Action space is the start and span of the radar waveform
    self.action_space = gym.spaces.Box(
        low=0, high=1, shape=(2,), dtype=np.float32)

    # Note: Assuming time axis is discretized based on the PRF. This doesn't work if the PRF changes, but by that point I hope to be working on real-world data
    self.freq_axis = np.linspace(
        0, self.channel_bw, self.observation_space.shape[1])
    self.time_axis = np.arange(0, self.spectrogram_duration, self.pri)

  def step(self, action: np.ndarray):
    # Update the radar waveform
    start_freq = action[0] * self.channel_bw
    radar_bw = min(action[1] * self.channel_bw, self.channel_bw - start_freq)

    # Update the communications interference
    self.spectrogram[:, :, 0] = self.interference.update_spectrogram(
        spectrogram=self.spectrogram[:, :, 0],
        freq_axis=self.freq_axis,
        start_time=self.time)
    
    # Update the radar observation
    n_freq_bins = np.digitize(
        radar_bw, self.freq_axis - np.min(self.freq_axis))
    n_time_bins_pulse = np.digitize(self.pulsewidth, self.time_axis)
    i_start_freq = np.digitize(
        start_freq, self.freq_axis - np.min(self.freq_axis), right=True)
    i_stop_freq = i_start_freq + n_freq_bins
    self.spectrogram[:, :, 1] = np.roll(self.spectrogram[:, :, 1], -1, axis=0)
    self.spectrogram[-1:, :, 1] = 0
    self.spectrogram[-n_time_bins_pulse:, i_start_freq:i_stop_freq, 1] = 255

    self.time += self.pri

    # Compute reward
    # TODO: Using simple reward function for now
    occupancy_reward = radar_bw / self.channel_bw
    collision_reward = np.sum(np.logical_and(self.spectrogram[-1:, :, 0] == 255,
                                    self.spectrogram[-1:, :, 1] == 255)) / n_freq_bins
    reward = occupancy_reward + collision_reward

    obs = self.spectrogram
    terminated = False
    truncated = False
    info = {}
    return obs, reward, terminated, truncated, info

  def reset(self, seed: int = None, options: dict = None):
    # Reset the spectrogram and time counter
    self.spectrogram = np.zeros(self.observation_space.shape, dtype=np.uint8)
    self.time = 0

    # Run the interference for a variable number of steps
    self.interference.reset()
    n_warmup_steps = self.np_random.integers(0, self.spectrogram.shape[0])
    for i in range(1, n_warmup_steps + 1):
      self.spectrogram[:, :, 0] = self.interference.update_spectrogram(
          spectrogram=self.spectrogram[:, :, 0],
          freq_axis=self.freq_axis,
          start_time=(i-1)*self.pri)
    self.interference.last_update_time = 0

    observation = self.spectrogram
    info = {}

    return observation, info

  def render(self):
    raise NotImplementedError

  def close(self):
    raise NotImplementedError


if __name__ == '__main__':
  env = SpectrumHopper()
  obs, info = env.reset()
  print(env.action_space)
  print(env.observation_space)

  for i in range(1):
    start_freq = np.random.uniform(0, 1)
    bandwidth = np.random.uniform(0, 1)
    obs, reward, term, trunc, info = env.step([start_freq, bandwidth])

  plt.figure()
  plt.imshow(obs[:, :, 0],
             extent=(env.freq_axis[0],
             env.freq_axis[-1], env.time_axis[0]*1e3, env.time_axis[-1]*1e3),
             aspect='auto')
  plt.colorbar()
  plt.show()
