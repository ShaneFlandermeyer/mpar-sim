from gymnasium.envs.registration import register
from mpar_sim.envs.particle_surveillance import ParticleSurveillance
from mpar_sim.envs.spectrum_hopper import SpectrumHopper

register(
    id="mpar_sim/ParticleSurveillance-v0",
    entry_point="mpar_sim.envs:ParticleSurveillance",
)

register(
    id="mpar_sim/SpectrumHopper-v0",
    entry_point="mpar_sim.envs:SpectrumHopper",
)
