from gymnasium.envs.registration import register
# from mpar_sim.envs.particle_surveillance import ParticleSurveillance
# from mpar_sim.envs.spectrum_hopper_2d import SpectrumHopper2D
from mpar_sim.envs.spectrum_hopper_1d import SpectrumHopper1D
from mpar_sim.envs.spectrum_hopper_recorded import SpectrumHopperRecorded

# register(
#     id="mpar_sim/ParticleSurveillance-v0",
#     entry_point="mpar_sim.envs:ParticleSurveillance",
# )

# register(
#     id="mpar_sim/SpectrumHopper2D-v0",
#     entry_point="mpar_sim.envs:SpectrumHopper2D",
# )

register(
    id="mpar_sim/SpectrumHopperRecorded-v0",
    entry_point="mpar_sim.envs:SpectrumHopperRecorded",
)

register(
    id="mpar_sim/SpectrumHopper1D-v0",
    entry_point="mpar_sim.envs:SpectrumHopper1D",
)

