from gymnasium.envs.registration import register
from mpar_sim.envs.spectrum_env import SpectrumEnv

register(
    id="mpar_sim/SpectrumEnv",
    entry_point=SpectrumEnv,
)