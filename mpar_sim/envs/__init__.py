from gymnasium.envs.registration import register
from mpar_sim.envs.simple_surveillance import SimpleParticleSurveillance

register(
  id="mpar_sim/SimpleParticleSurveillance-v0",
  entry_point="mpar_sim.envs:SimpleParticleSurveillance",
)