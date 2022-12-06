from gymnasium.envs.registration import register
from mpar_sim.envs.surveillance_env import ParticleSurveillance

register(
  id="mpar_sim/ParticleSurveillance-v0",
  entry_point="mpar_sim.envs:ParticleSurveillance",
  max_episode_steps=300,
)