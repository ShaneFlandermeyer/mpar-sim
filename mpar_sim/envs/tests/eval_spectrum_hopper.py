import os
import numpy as np
from ray import tune
from ray.rllib.algorithms.algorithm import Algorithm
from mpar_sim.envs.spectrum_hopper import SpectrumHopper
from mpar_sim.models.networks.rnn import LSTMActorCritic
from ray.rllib.models import ModelCatalog



tune.register_env(
      "SpectrumHopper", lambda env_config: SpectrumHopper(env_config))
ModelCatalog.register_custom_model("LSTM", LSTMActorCritic)

results_path = "/home/shane/onedrive/research/my_stuff/trs23/data/spectrum_metrics"

# Need the tuner for the env config params
tuner = tune.Tuner.restore(results_path)
results_grid = tuner.get_results()
result = results_grid[1]
algo = Algorithm.from_checkpoint(result.checkpoint)
config = result.config
  
# Prepare env
env_config = config["env_config"]
print(env_config)
# env_config["render_mode"] = "human"
env = SpectrumHopper(env_config)
obs, info = env.reset()
done = False
total_reward = 0
# Initialize memory
lstm_cell_size = config["model"]["lstm_cell_size"]
init_state = state = [
    np.zeros([lstm_cell_size], np.float32) for _ in range(4)]
prev_action = np.zeros(env.action_space.shape, np.float32)
prev_reward = 0
  
while not done:
  # action = env.action_space.sample()
  action, state, _ = algo.compute_single_action(
      obs, state, prev_action=prev_action, prev_reward=prev_reward, explore=False)
  obs, reward, terminated, truncated, info = env.step(action)
  done = terminated or truncated
  prev_action = action
  prev_reward = reward
  total_reward += reward
  # env.render()
print("Total eval. reward:", total_reward)