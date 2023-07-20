# %%
import matplotlib.pyplot as plt
import numpy as np
import pytest

from mpar_sim.models.measurement import LinearMeasurementModel
from mpar_sim.models.transition import ConstantVelocity
from mpar_sim.tracking import JPDATracker, KalmanFilter
from mpar_sim.types import FalseDetection, Track, Trajectory, TrueDetection

def test_jpda():
  # %% [markdown]
  # ## Simulate Ground Truth

  # %%
  n_steps = 20
  current_time = last_update = 0
  dt = 1
  seed = 0
  np.random.seed(seed)

  paths = []

  # Target 1
  path = Trajectory()
  path.append(state=np.array([0, 1, 0, 1]),
              covar=np.diag([1.5, 0.5, 1.5, 0.5]),
              timestamp=current_time)
  transition_model = ConstantVelocity(ndim_pos=2,
                                      noise_diff_coeff=0.005,
                                      seed=seed)
  for i in range(n_steps):
    current_time += dt
    state = transition_model(path.state, dt=dt, noise=True)
    path.append(state=state, timestamp=current_time)
  paths.append(path)

  # Target 2
  current_time = last_update = 0
  path = Trajectory()
  path.append(state=np.array([0, 1, 20, -1]),
              covar=np.diag([1.5, 0.5, 1.5, 0.5]),
              timestamp=current_time)
  for i in range(n_steps):
    current_time += dt
    state = transition_model(path.state, dt=dt, noise=True)
    path.append(state=state, timestamp=current_time)
  paths.append(path)

  # Plot the ground truth
  states = [np.stack([state.state for state in path]).T for path in paths]
  plt.figure()
  plt.plot(states[0][0], states[0][2], '--', label='Target 1')
  plt.plot(states[1][0], states[1][2], '--', label='Target 2')
  plt.grid()


  # %% [markdown]
  # ## Simulate Measurements

  # %%
  pd = 0.9
  measurement_model = LinearMeasurementModel(
      ndim_state=4,
      covar=np.diag([0.75, 0.75]),
      measured_dims=[0, 2],
      seed=seed,
  )

  all_detections = []
  detections = []
  for i in range(n_steps):
    current_detections = []
    for path in paths:
      state = path[i]
      if np.random.rand() <= pd:
        measurement = measurement_model(state.state, noise=True)
        detection = TrueDetection(measurement=measurement,
                                  measurement_model=measurement_model,
                                  timestamp=state.timestamp,
                                  origin=state)
        current_detections.append(detection)

      true_x = state.state[0]
      true_y = state.state[2]
      for _ in range(np.random.randint(5)):
        x = np.random.uniform(true_x - 10, true_x + 10)
        y = np.random.uniform(true_y - 10, true_y + 10)
        detection = FalseDetection(
            measurement=np.array([x, y]),
            timestamp=state.timestamp,
        )
        current_detections.append(detection)
    detections.append(current_detections)
    all_detections.extend(current_detections)


  # %%
  true_states = np.array([[state.state for state in path] for path in paths])
  measurements = np.array(
      [detection.measurement for detection in all_detections])

  # %% [markdown]
  # ## Track Processing

  # %%
  
  # Initialize filters
  kf = KalmanFilter(
      transition_model=transition_model,
      measurement_model=measurement_model)
  tracks = []
  for path in paths:
    track = Track(
      history=[path[0]],
      filter=kf,
    )
    tracks.append(track)    
  jpda = JPDATracker(
      tracks=tracks,
      pd=pd,
      pg=0.99
  )
    
  # Sequentially process detections
  current_time = last_update = 0
  for i, current_detections in enumerate(detections):
    current_time = current_detections[0].timestamp
    measurements = [d.measurement for d in current_detections]
    dt = current_time - last_update
    pred_states, pred_covars = jpda.predict(dt=dt)
    states, covars = jpda.update(measurements=measurements, 
                                 predicted_states=pred_states,
                                 predicted_covars=pred_covars)
    # Update tracks
    for j, track in enumerate(tracks):
      track.append(
          state=states[j],
          covar=covars[j],
          timestamp=current_time,
      )
    last_update = current_time
  
  for i, track in enumerate(tracks):
    true_states = np.array([state.state for state in paths[i]])
    track_pos = np.array([state.state[[0, 2]] for state in track])
    track_vel = np.array([state.state[[1, 3]] for state in track])
    pos_mse = np.mean(np.linalg.norm(true_states[:, [0, 2]] - track_pos, axis=1))
    vel_mse = np.mean(np.linalg.norm(true_states[:, [1, 3]] - track_vel, axis=1))
    
    assert pos_mse < 2.0
    assert vel_mse < 0.3
    
if __name__ == '__main__':
  # test_jpda()
  pytest.main()