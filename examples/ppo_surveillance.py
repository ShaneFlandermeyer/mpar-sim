# %%
# Imports
import numpy as np
from stonesoup.models.transition.linear import (
    CombinedLinearGaussianTransitionModel, ConstantVelocity, SingerApproximate)
from stonesoup.types.array import CovarianceMatrix, StateVector
from stonesoup.types.state import GaussianState
from mpar_sim.defaults import default_radar, default_raster_scan_agent, default_gbest_pso, default_lbest_pso
import mpar_sim.envs
import gymnasium as gym

# %%
# Agent object definition
class PPOSurveillance():
    """
    TODO: Implement me
    """

# %%
# Set up the environment
# Target generation model
transition_model = CombinedLinearGaussianTransitionModel([
    ConstantVelocity(10),
    ConstantVelocity(10),
    ConstantVelocity(0),
])
initial_state_mean = StateVector([10e3, 10, 0, 0, 0, 0])
initial_state_covariance = CovarianceMatrix(
    np.diag([1000, 100, 1000, 100, 1000, 100]))
initial_state = GaussianState(
    initial_state_mean, initial_state_covariance)
# Radar system object
radar = default_radar()
radar.false_alarm_rate = 1e-7
radar.include_false_alarms = False
# Environment
env = gym.make('mpar_sim/ParticleSurveillance-v0',
               radar=radar,
               transition_model=transition_model,
               initial_state=initial_state,
               birth_rate=0.01,
               death_probability=0,
               initial_number_targets=20,
               render_mode='human',
               )

# %% 
# Create the agent and run the simulation
agent = default_raster_scan_agent()

obs, info = env.reset()
for i in range(1000):
  look = agent.act(env.time)[0]
  obs, reward, terminated, truncated, info = env.step(
      dict(
          azimuth_steering_angle=look.azimuth_steering_angle,
          elevation_steering_angle=look.elevation_steering_angle,
          azimuth_beamwidth=look.azimuth_beamwidth,
          elevation_beamwidth=look.elevation_beamwidth,
          bandwidth=look.bandwidth,
          pulsewidth=look.pulsewidth,
          prf=look.prf,
          n_pulses=look.n_pulses,
          tx_power=look.tx_power,
      )
  )

# %%
# Visualizations
from pyswarms.utils.functions import single_obj as fx
from pyswarms.utils.plotters import (plot_cost_history, plot_contour, plot_surface)
from pyswarms.utils.plotters.formatters import Mesher
from pyswarms.utils.plotters.formatters import Designer
import matplotlib.pyplot as plt

d = Designer(limits=[(-45,45), (-45,45)], label=['azimuth (deg.)', 'elevation (deg.)'])
animation = plot_contour(pos_history=env.swarm_optim.pos_history[::2],
                        designer=d,)
animation.save('/home/shane/particles.gif', writer='ffmpeg', 
              fps=10)                   
# %%
from stonesoup.plotter import Plotter, Dimension


plotter = Plotter(Dimension.THREE)
plotter.plot_sensors(radar, "Radar")
plotter.plot_ground_truths(env.target_history, radar.position_mapping)
plotter.plot_measurements(env.detection_history, radar.position_mapping)
plt.show()
# %%
