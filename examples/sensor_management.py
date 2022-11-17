# %% [markdown]
# # Sensor Management Example

# %%
import numpy as np
import random
from ordered_set import OrderedSet
from datetime import datetime, timedelta
from mpar_sim.radar import PhasedArrayRadar


start_time = datetime.now()

from stonesoup.models.transition.linear import CombinedLinearGaussianTransitionModel, ConstantVelocity
from stonesoup.types.groundtruth import GroundTruthPath, GroundTruthState

# %% [markdown]
# ## Generate Ground Truths

# %%
np.random.seed(1990)
random.seed(1990)

# Generate transition model
transition_model = CombinedLinearGaussianTransitionModel([
    ConstantVelocity(0.005),
    ConstantVelocity(0.005),
    ConstantVelocity(0.005),
])

yps = range(0, 100, 10)  # y value for prior state
truths = OrderedSet()
ntruths = 3
time_max = 50

xdirection = 1
ydirection = 1
zdirection = 0

# Generate ground truths
for i in range(ntruths):
  truth = GroundTruthPath([
      GroundTruthState([0, xdirection, yps[i], ydirection, 0, zdirection],
                       timestamp=start_time)],
      id=f"id{i}")
  # TODO: Create a radar ground truth class with an RCS field
  truth.rcs = 10
  for j in range(1, time_max):
    current_truth = GroundTruthState(transition_model.function(truth[j-1], noise=True, time_interval=timedelta(seconds=1)),
                         timestamp=start_time + timedelta(seconds=j))
    current_truth.rcs = 10
    truth.append(current_truth)
    
  truths.add(truth)
  
  # Alternate directions when initiating tracks
  xdirection *= -1
  if i % 2 == 0:
    ydirection *= -1

# %% [markdown]
# Plot ground truths using the Plotterly class

# %%
from stonesoup.plotter import Plotterly

plotter = Plotterly()
plotter.plot_ground_truths(truths, [0, 2])
plotter.fig

# %% [markdown]
# ## Create Sensor

# %%
from mpar_sim.radar import PhasedArrayRadar
from stonesoup.types.state import StateVector

sensor = PhasedArrayRadar(
  position=np.array([10, 0, 0]),
  rotation_offset=np.array([0, 0, 180])
)
sensor.timestamp = start_time

# %% [markdown]
# ## Create the Kalman Predictor and Updater

# %%
from stonesoup.predictor.kalman import KalmanPredictor
predictor = KalmanPredictor(transition_model)

from stonesoup.updater.kalman import ExtendedKalmanUpdater
updater = ExtendedKalmanUpdater(measurement_model=None)
# Measurement model is added to detections by the sensor

# %% [markdown]
# ## Run the Kalman Filters
# 
# Create a Kalman filter prior for each target. Here, each prior is offset by 0.5 in the y-direction, so the position of the track is initially not very accurate. The velocity is also offset by 0.5 in the x and y directions

# %%
from stonesoup.types.state import GaussianState

priors = []
xdirection = 1.2
ydirection = 1.2
for i in range(ntruths):
  priors.append(GaussianState([0, xdirection, yps[i]+0.2, ydirection, 0, 0],
                              np.diag([0.5, 0.5, 0.5, 0.5, 0, 0] + np.random.normal(0, 5e-4, 6)),
                              timestamp=start_time))
  xdirection *= -1
  if i % 2 == 0:
    ydirection *= -1

# %% [markdown]
# Initialize the tracks by creating an empty list and appending the priors generated.

# %%
from stonesoup.types.track import Track

tracks = {Track([prior]) for prior in priors}
tracks

# %% [markdown]
# ## Create sensor manager

# %%
# TODO: Create a sensor manager class

# %% [markdown]
# ## Run the sensor managers
# 
# First, create a hypothesizer and data associator

# %%
from stonesoup.hypothesiser.distance import DistanceHypothesiser
from stonesoup.measures import Mahalanobis
hypothesiser = DistanceHypothesiser(
    predictor, updater, measure=Mahalanobis(), missed_distance=5)

from stonesoup.dataassociator.neighbour import GNNWith2DAssignment
data_associator = GNNWith2DAssignment(hypothesiser)

# %%
# Generate list of timesteps from ground truth timestamps
from mpar_sim.beam.common import aperture2beamwidth
from mpar_sim.look import RadarLook


timesteps = []
for state in truths[0]:
  timesteps.append(state.timestamp)

for timestep in timesteps[1:]:
  # TODO: Choose actions
  look = RadarLook(
      start_time=start_time,
      tx_power=2560,
      azimuth_steering_angle=0,
      elevation_steering_angle=0,
      azimuth_beamwidth=aperture2beamwidth(
          sensor.element_spacing*sensor.n_elements_x, sensor.wavelength),
      elevation_beamwidth=aperture2beamwidth(
          sensor.element_spacing*sensor.n_elements_y, sensor.wavelength),
      bandwidth=1e6,
      pulsewidth=1e-6,
      prf=5e3,
      n_pulses=10,
  )

  # TODO: Schedule actions
  sensor.load_look(look)

  measurements = set()

  measurements |= sensor.measure(OrderedSet(
      truth[timestep] for truth in truths), noise=True)

  # Associate measurements to tracks
  hypotheses = data_associator.associate(tracks, measurements, timestep)

  # Update tracks
  for track in tracks:
    hypothesis = hypotheses[track]
    if hypothesis.measurement:
      post = updater.update(hypothesis)
      track.append(post)
    else:
      track.append(hypothesis.prediction)


# %% [markdown]
# Plot the ground truth, tracks, and uncertainty ellipses for each target

# %%
plotter = Plotterly()
plotter.plot_sensors(sensor)
plotter.plot_ground_truths(truths, [0, 2])
plotter.plot_tracks(tracks, [0, 2], uncertainty=True)
plotter.fig


