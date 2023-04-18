import gymnasium as gym
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import VecMonitor
import mpar_sim.envs
import torch.nn as nn

# %%


def make_env(env_id, idx):
  def thunk():
    if idx == -1:
      render_mode = "human"
    else:
      render_mode = "rgb_array"
    env = gym.make(env_id,
                    filename='/home/shane/data/HOCAE_Snaps_bool_cleaned.dat',
                    channel_bandwidth=100e6,
                    fft_size=1024,
                    render_mode=render_mode)
    env = gym.wrappers.TimeLimit(env, max_episode_steps=1000)
    # env = gym.wrappers.ClipAction(env)
    return env
  return thunk

if __name__ == '__main__':
    env_id = "mpar_sim/SpectrumHopper1D-v0"
    num_cpu = 1  # Number of processes to use
    # Create the vectorized environment
    env = DummyVecEnv([make_env(env_id, i) for i in range(num_cpu)])
    env = VecMonitor(env, filename='./results.txt', info_keywords=("widest","collisions"))
    model = SAC("MlpPolicy", env, verbose=1, tensorboard_log="./log/", gamma=0)
    model.learn(total_timesteps=2_000_000, tb_log_name="spectrum_hopper")