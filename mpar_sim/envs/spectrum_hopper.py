import gymnasium as gym
import numpy as np


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
    self.interference_bw = 20e6
    self.interference_duration = 5e-3
    self.spectrogram_duration = 30e-3
    self.pri = 409.6e-6
    self.pulsewidth = 10e-6

    # TODO: Add the observation space

    # Action space is the start and span of the radar waveform
    self.action_space = gym.spaces.Box(
        low=np.array([0, 0], dtype=np.float32),
        high=np.array([1, 1], dtype=np.float32),
    )

  def step(self, action: np.ndarray):
    # Update the radar waveform
    start_freq = action[0] * self.channel_bw
    radar_bw = min(action[1] * self.channel_bw, self.channel_bw - start_freq)

    # TODO: Update the radar observation
    radar_dx = int(
        max(radar_bw / self.channel_bw * self.spectrogram.shape[1], 1)
    )
    radar_dy = int(
        max(self.pulsewidth / self.spectrogram_duration *
            self.spectrogram.shape[0], 1)
    )

    # TODO: Update the communications interference
    # TODO: Handle the fact that the PRI and interference changes are not time-aligned
    if self.time > self.interference_duration:
      self._update_interference()

    # TODO: Update the interference observation

    # TODO: Compute reward

    self.time += self.pri

    raise NotImplementedError

  def reset(self, seed: int = None, options: dict = None):
    # For the spectrogram, the first channel is the interference occupancy map, and the second is for the radar
    self.spectrogram = np.zeros((84, 84, 2), dtype=np.uint8)
    self.interference_start_freq = self.np_random.uniform(0, self.channel_bw)
    self.interference_sweep_direction = self.np_random.choice([-1, 1])
    self.time = 0

    observation = self.spectrogram
    info = {}

    return observation, info

  def render(self):
    raise NotImplementedError

  def close(self):
    raise NotImplementedError

  ##########################
  # Internal helper methods
  ##########################
  def _update_interference(self):
    next_start_freq = self.interference_start_freq + \
        self.interference_sweep_direction * self.interference_bw
    if next_start_freq < 0 or next_start_freq > self.channel_bw:
      self.interference_sweep_direction *= -1
      self.interference_start_freq += self.interference_sweep_direction * self.interference_bw
    else:
      self.interference_start_freq = next_start_freq


if __name__ == '__main__':
  env = SpectrumHopper()
  obs, info = env.reset()
  env.step([0.5, 0.5])
