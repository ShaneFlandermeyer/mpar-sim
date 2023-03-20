from gymnasium.envs.registration import register
from mpar_sim.envs.particle_surveillance import ParticleSurveillance
from mpar_sim.envs.spectrum_hopper import SpectrumHopper
from mpar_sim.envs.spectrum_hopper_1d import SpectrumHopper1D

register(
    id="mpar_sim/ParticleSurveillance-v0",
    entry_point="mpar_sim.envs:ParticleSurveillance",
)

register(
    id="mpar_sim/SpectrumHopper-v0",
    entry_point="mpar_sim.envs:SpectrumHopper",
)

register(
    id="mpar_sim/SpectrumHopper1D-v0",
    entry_point="mpar_sim.envs:SpectrumHopper1D",
)

