from gymnasium.envs.registration import register
from mpar_sim.envs.particle_surveillance import ParticleSurveillance

register(
  id="mpar_sim/ParticleSurveillance-v0",
  entry_point="mpar_sim.envs:ParticleSurveillance",
)
