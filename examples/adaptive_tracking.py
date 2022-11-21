# %% [markdown]
# # Adaptive Tracking Simulation Example

# %% [markdown]
# ## Setup

# %%
from datetime import datetime, timedelta
start_time = datetime.now()

# %% [markdown]
# ## Radar system

# %%
from mpar_sim.radar import PhasedArrayRadar
from mpar_sim.beam.beam import RectangularBeam, GaussianBeam
from mpar_sim.looks.look import Look
from mpar_sim.resource_management import PAPResourceManager
from mpar_sim.schedulers import BestFirstScheduler
import numpy as np

radar = PhasedArrayRadar(
    position=np.array([[0], [0], [0]]),
    position_mapping=(0, 2, 4),
    rotation_offset=np.array([[0], [0], [0]]),
    # Array parameters
    n_elements_x=16,
    n_elements_y=16,
    element_spacing=0.5,  # Wavelengths
    element_tx_power=10,
    # System parameters
    center_frequency=3e9,
    system_temperature=290,
    noise_figure=4,
    # Scan settings
    beam_shape=GaussianBeam,
    az_fov=[-90, 90],
    el_fov=[-90, 90],
    # Detection settings
    false_alarm_rate=1e-6,
)
radar.timestamp = start_time

manager = PAPResourceManager(radar,
                             max_duty_cycle=0.1,
                             max_bandwidth=100e6)
scheduler = BestFirstScheduler(manager,
                               sort_key="start_time",
                               reverse_sort=False,)


# %% [markdown]
# Raster scan agent

# %%
from mpar_sim.agents.raster_scan import RasterScanAgent
import numpy as np

search_agent = RasterScanAgent(
    azimuth_scan_limits=np.array([-30, 30]),
    elevation_scan_limits=np.array([-5, 5]),
    azimuth_beam_spacing=0.8,
    elevation_beam_spacing=0.8,
    azimuth_beamwidth=7.5,
    elevation_beamwidth=7.5,
    bandwidth=100e6,
    pulsewidth=1e-6,
    prf=5e3,
    n_pulses=10,
)

# %% [markdown]
# ## Tracker Components

# %% [markdown]
# Create tracker

# %%
from stonesoup.measures import Mahalanobis, Euclidean
from stonesoup.hypothesiser.distance import DistanceHypothesiser
from stonesoup.updater.kalman import ExtendedKalmanUpdater
from stonesoup.predictor.kalman import KalmanPredictor
from stonesoup.models.transition.linear import CombinedLinearGaussianTransitionModel, ConstantVelocity
from stonesoup.gater.distance import DistanceGater

# KF prediction model. Assuming it's matched to the true target model for now
transition_model = CombinedLinearGaussianTransitionModel([
    ConstantVelocity(10),
    ConstantVelocity(10),
    ConstantVelocity(0.0),
])
predictor = KalmanPredictor(transition_model)

updater = ExtendedKalmanUpdater(measurement_model=None)

hypothesizer = DistanceHypothesiser(
    predictor, updater, measure=Mahalanobis(), missed_distance=100)
gater = DistanceGater(hypothesizer, measure=Mahalanobis(), gate_threshold=25)


# %% [markdown]
# Create the data associator

# %%
from stonesoup.dataassociator.neighbour import GNNWith2DAssignment
data_associator = GNNWith2DAssignment(gater)

# %% [markdown]
# Create the deleter

# %%
from stonesoup.deleter.time import UpdateTimeStepsDeleter, UpdateTimeDeleter
deleter = UpdateTimeDeleter(timedelta(seconds=2))
# deleter = UpdateTimeStepsDeleter(10)

# %% [markdown]
# Create the initiator

# %%
from stonesoup.types.state import GaussianState
from stonesoup.initiator.simple import MultiMeasurementInitiator, SimpleMeasurementInitiator
import numpy as np
from mpar_sim.initiator.initators import MofNInitiator


initiator = MofNInitiator(
    prior_state=GaussianState([0, 0, 0, 0, 0, 0], np.diag([0, 0, 0, 0, 0, 0])),
    measurement_model=None,
    deleter=deleter,
    data_associator=data_associator,
    updater=updater,
    confirmation_threshold=[3,5],
)

# %% [markdown]
# Tracking agent

# %%
from mpar_sim.agents.track_while_scan import TWSAgent
from mpar_sim.agents.adaptive_track import AdaptiveTrackAgent

track_agent = AdaptiveTrackAgent(
    initiator,
    data_associator,
    predictor,
    updater,
    deleter,
    # Adaptive track parameters 
    track_sharpness=0.15,
    min_revisit_rate=0.5,
    max_revisit_rate=5,
    confirm_rate=20,
    # Task parameters
    azimuth_beamwidth=1.5,
    elevation_beamwidth=1.5,
    bandwidth=100e6,
    pulsewidth=1e-6,
    prf=5e3,
    n_pulses=100
)

# track_agent = TWSAgent(initiator, data_associator, updater, deleter)

# %% [markdown]
# ## Run the simulation

# %%
import operator
from stonesoup.types.groundtruth import GroundTruthPath, GroundTruthState
from stonesoup.types.array import CovarianceMatrix
from stonesoup.types.state import StateVector
import random
# Set the simulation seed
seed = np.random.randint(0, 2**32-1)
np.random.seed(seed)
random.seed(seed)

# Simulation-level parameters
n_steps = 500
include_noise = True

# Target generation parameters
n_targets_max = 20
initial_state_mean = StateVector([100, 10, 0, 0, 0, 0])
initial_state_covariance = CovarianceMatrix(np.diag([100, 5, 100, 5, 0, 0]))
initial_state = GaussianState(initial_state_mean, initial_state_covariance)
death_probability = 0.0
birth_probability = 0.1
target_rcs = 1


truths = []
all_truths = []
confirmed_tracks = set()
tentative_tracks = set()
all_measurements = []
all_tracks = set()

# Simulation time variables. 
time = start_time
scheduler_time = start_time
for istep in range(n_steps):
  detections = set()

  ########################################
  # Target birth/death
  ########################################
  # Delete targets according to the death process
  truths = [truth for truth in truths if np.random.rand() > death_probability]
  # Also randomly delete targets if we have exceeded the maximum target count
  if len(truths) > n_targets_max:
    indices = np.random.choice(
        len(truths), len(truths) - n_targets_max, replace=False)
    for index in sorted(indices, reverse=True):
      del truths[index]
    

  # Targets, be reborn!
  for _ in range(np.random.poisson(birth_probability)):
    
    if len(truths) >= n_targets_max:
      break
    
    # Sample an initial state from the mean and covariance defined above
    state_vector = initial_state.state_vector + \
        initial_state.covar @ np.random.randn(initial_state.ndim, 1)
    state = GroundTruthState(
        state_vector=state_vector,
        timestamp=time,
    )
    # Give the target an RCS
    # TODO: Create a GroundTruthTarget class with an RCS attribute
    state.rcs = target_rcs
    # Add to the list of truths
    truth = GroundTruthPath([state])
    truths.append(truth)
    all_truths.append(truth)

  ########################################
  # Allocate resources and simulate
  ########################################
  # Generate looks from each agent
  search_look = search_agent.act(current_time=scheduler_time)
  track_looks = track_agent.act(current_time=scheduler_time)
  looks = [search_look] + track_looks

  # Schedule new looks, sorted so that the task with the nearest end time is selected first below
  scheduler.schedule(looks, scheduler_time)
  manager.allocated_tasks.sort(
    key=operator.attrgetter("end_time"), reverse=True)
  # Minimum start time of all scheduled tasks. When this changes, a new "batch" of tasks has been allocated and the scenario needs to be updated
  min_start_time = min([task.start_time for task in manager.allocated_tasks]
                 ) if manager.allocated_tasks else time

  # Get the next look and simulate it
  look = manager.allocated_tasks.pop()
  scheduler_time = look.end_time
  radar.load_look(look)
  detections = radar.measure(truths, noise=include_noise)
  
  # Update tracks
  confirmed_tracks = track_agent.update_tracks(detections, time)

  # Update targets
  if min_start_time > time:
    dt = min_start_time - time
    time = min_start_time
    # Update targets
    for truth in truths:
      truth.append(GroundTruthState(
          transition_model.function(truth[-1],
                                    noise=include_noise,
                                    time_interval=dt),
          timestamp=time))
      truth[-1].rcs = target_rcs
            
  all_measurements.append(detections)
  all_tracks |= confirmed_tracks


# %% [markdown]
# ## Plot simulation results

# %%
from stonesoup.plotter import Plotterly

plotter = Plotterly()
plotter.plot_sensors(radar, "Radar")
plotter.plot_ground_truths(all_truths, [0, 2])
plotter.plot_measurements(all_measurements, [0, 2])
plotter.plot_tracks(all_tracks, [0,2])

plotter.fig


