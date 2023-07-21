
from matplotlib import pyplot as plt
import numpy as np
from mpar_sim.models.transition import ConstantVelocity
from mpar_sim.models.measurement.linear import LinearMeasurementModel
from mpar_sim.tracking.kalman import KalmanFilter
from mpar_sim.tracking.pda import PDAFilter
from mpar_sim.types.detection import FalseDetection, TrueDetection
from mpar_sim.types.track import Track

from mpar_sim.types.trajectory import Trajectory
import numpy as np
from scipy.stats import uniform
import pytest


def test_pda():
  # Generate ground truth
  n_steps = 50
  current_time = last_update = 0
  dt = 1
  seed = 0
  np.random.seed(seed)

  trajectory = Trajectory()
  trajectory.append(state=np.array([0, 1, 0, 1]),
                    covar=np.diag([1.5, 0.5, 1.5, 0.5]),
                    timestamp=current_time)
  transition_model = ConstantVelocity(ndim_pos=2,
                                      noise_diff_coeff=0.005,
                                      seed=seed)
  for i in range(n_steps):
    state = transition_model(trajectory.state, dt=dt, noise=True)
    timestamp = i*dt
    trajectory.append(state=state, timestamp=timestamp)

  # Generate measurements
  measurement_model = LinearMeasurementModel(
      ndim_state=4,
      covar=np.array([[0.75, 0], [0, 0.75]]),
      measured_dims=[0, 2],
      seed=seed,
  )
  pd = 0.9
  detections = []
  track_pos = []
  track_vel = []
  for i, state in enumerate(trajectory):

    current_detections = []
    if np.random.rand() < pd:
      measurement = measurement_model(state.state, noise=True)
      detection = TrueDetection(measurement=measurement,
                                measurement_model=measurement_model,
                                timestamp=state.timestamp,
                                origin=state)
      current_detections.append(detection)

    # Add clutter
    true_x = state.state[0]
    true_y = state.state[2]
    for _ in range(np.random.randint(10)):
      x = uniform.rvs(true_x - 10, 20)
      y = uniform.rvs(true_y - 10, 20)
      detection = FalseDetection(
          measurement=np.array([x, y]),
          timestamp=state.timestamp,
      )
      current_detections.append(detection)
      
    detections.append(current_detections)


  # Run tracker
  kf = KalmanFilter(
      transition_model=transition_model,
      measurement_model=measurement_model,
  )
  pda = PDAFilter(
      filter=kf,
      pd=pd,
      pg=0.95,
      # clutter_density=0.125,
  )
  track = Track(
    history=[trajectory[0]],
    filter=pda)
  for i, current_detections in enumerate(detections):
    current_time = current_detections[0].timestamp
    measurements = [d.measurement for d in current_detections]
    dt = current_time - last_update
    pred_state, pred_covar = track.predict(dt=dt)
    state, covar = track.update(measurements=measurements,
                                predicted_state=pred_state,
                                predicted_covar=pred_covar)
    track.append(state=state, covar=covar, timestamp=current_time)
    last_update = current_time
    
  true_pos = np.array([state.state[[0, 2]] for state in trajectory])
  true_vel = np.array([state.state[[1, 3]] for state in trajectory])
  track_pos = np.array([state.state[[0, 2]] for state in track[1:]])
  track_vel = np.array([state.state[[1, 3]] for state in track[1:]])

  # true_states = np.array([state.state for state in trajectory])
  pos_mse = np.mean(np.linalg.norm(true_pos - track_pos, axis=1))
  vel_mse = np.mean(np.linalg.norm(true_vel - track_vel, axis=1))
  assert pos_mse < 1.0
  assert vel_mse < 0.5


if __name__ == '__main__':
  pytest.main()
