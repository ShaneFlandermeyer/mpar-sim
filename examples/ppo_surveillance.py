# %%
# Imports
import time
from stonesoup.plotter import Plotter, Dimension
import matplotlib.pyplot as plt
from pyswarms.utils.plotters.formatters import Designer
from pyswarms.utils.plotters.formatters import Mesher
from pyswarms.utils.plotters import (
    plot_cost_history, plot_contour, plot_surface)
from pyswarms.utils.functions import single_obj as fx
import numpy as np
from stonesoup.models.transition.linear import (
    CombinedLinearGaussianTransitionModel, ConstantVelocity, SingerApproximate)
from stonesoup.types.array import CovarianceMatrix, StateVector
from stonesoup.types.state import GaussianState
from mpar_sim.defaults import default_radar, default_raster_scan_agent, default_gbest_pso, default_lbest_pso, default_scheduler
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
    ConstantVelocity(10),
])

# NOTE: Specifying initial state in terms of az/el/range (in degrees)!
initial_state = GaussianState(
    state_vector=[0, 0, 0, 0, 15e3, 0],
    covar=np.diag([10, 100, 10, 100, 5e3, 100])
)
# Radar system object
radar = default_radar()
radar.false_alarm_rate = 1e-7
radar.include_false_alarms = False
radar.element_tx_power = 1000
radar.max_range = 20e3
scheduler = default_scheduler(radar)

# Environment
env = gym.make('mpar_sim/ParticleSurveillance-v0',
               radar=radar,
               transition_model=transition_model,
               initial_state=initial_state,
               birth_rate=0,
               death_probability=0,
               initial_number_targets=20,
               n_confirm_detections=1,
               randomize_initial_state=True,
               render_mode='rgb_array',
               )

# %%
# Create the agent and run the simulation
agent = default_raster_scan_agent()

obs, info = env.reset()
tic = time.time()
for i in range(1000):
  # Create a look and schedule it. This fills in the tx power field based on the number of elements used to form the beam
  action = agent.act(env.time)
  scheduler.schedule(list(action), env.time)
  look = scheduler.manager.allocated_tasks.pop()

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
  done = terminated or truncated
  if done:
    # At this point, you would normally reset the environment. For this demonstration, we just break out of the loop
    print("Episode finished after {} timesteps".format(i+1))
    break
    obs, info = env.reset()
toc = time.time()
print(f"Episode took {toc-tic} seconds")

# %%
# Visualizations

d = Designer(limits=[(-45, 45), (-45, 45)],
             label=['azimuth (deg.)', 'elevation (deg.)'])
animation = plot_contour(pos_history=env.swarm_optim.pos_history[::2],
                         designer=d,)
# animation.save('/home/shane/particles.gif', writer='ffmpeg',
#               fps=10)
# %%


plotter = Plotter(Dimension.THREE)
plotter.plot_sensors(radar, "Radar")
plotter.plot_ground_truths(env.target_history, radar.position_mapping)
plotter.plot_measurements(env.detection_history, radar.position_mapping)
plt.show()
# %%
