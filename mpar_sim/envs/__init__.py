from gymnasium.envs.registration import register
from mpar_sim.envs.spectrum_env import SpectrumEnv
from mpar_sim.envs.discrete_spectrum_env import DiscreteSpectrumEnv

register(
    id="mpar_sim/SpectrumEnv",
    entry_point=SpectrumEnv,
)

register(
    id="mpar_sim/DiscreteSpectrumEnv",
    entry_point=DiscreteSpectrumEnv,
)

