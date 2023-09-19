from typing import List, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pytest

from mpar_sim.models.measurement import LinearMeasurementModel
from mpar_sim.models.transition import ConstantVelocity
from mpar_sim.tracking.kalman import KalmanFilter
from mpar_sim.tracking.unscented_kalman import UnscentedKalmanFilter
from mpar_sim.types import FalseDetection, Track, Trajectory, TrueDetection
from mpar_sim.types.state import State
from scipy.stats import multivariate_normal
from mpar_sim.tracking.spa import spada, TotalSPA
from mpar_sim.tracking.gaussian import mix_gaussians


def generate_trajectories(seed=None):
  # Test SPA methods on a simple crossing target scenario
  n_steps = 200
  current_time = last_update = 0
  dt = 1

  # Generate crossing target trajectories
  paths = [Trajectory() for _ in range(2)]
  paths[0].append(state=np.array([0, 1, 0, 1]),
                  timestamp=current_time)
  paths[1].append(state=np.array([0, 1, 20, -1]),
                  timestamp=current_time)
  transition_model = ConstantVelocity(ndim_pos=2,
                                      q=0.005,
                                      seed=seed)
  for i in range(n_steps):
    current_time += dt
    paths[0].append(state=transition_model(paths[0].state, dt=dt, noise=True),
                    timestamp=current_time)
    paths[1].append(state=transition_model(paths[1].state, dt=dt, noise=True),
                    timestamp=current_time)

  return paths


def generate_measurements(paths, pd: float, mu_c: int, seed=None):
  n_steps = len(paths[0])
  mm = LinearMeasurementModel(
      ndim_state=4,
      covar=np.diag([0.75, 0.75]),
      measured_dims=[0, 2],
      seed=seed,
  )

  measurements = []
  unordered_measurements = []
  for i in range(n_steps):
    new_measurements = []
    # Object measurements
    for path in paths:
      state = path[i]
      if np.random.rand() < pd:
        measurement = mm(state.state, noise=True)
        detection = TrueDetection(measurement=measurement,
                                  timestamp=state.timestamp,
                                  origin=path)
        new_measurements.append(detection)

      # False measurements
      true_x, true_y = state.state[0], state.state[2]
      for _ in range(np.random.randint(mu_c)):
        x = np.random.uniform(true_x - 10, true_x + 10)
        y = np.random.uniform(true_y - 10, true_y + 10)
        Vc = (10 + 10) * (10 + 10)  # Clutter volume
        measurement = FalseDetection(
            measurement=np.array([x, y]),
            timestamp=state.timestamp,
        )
        new_measurements.append(measurement)
    measurements.append(new_measurements)
  return measurements


def test_spa():
  # Parameters
  pd = 0.9
  mu_c = 5

  seed = 0
  np.random.seed(seed)

  paths = generate_trajectories(seed=seed)
  # paths = [paths[1]]
  measurements = generate_measurements(paths, pd=pd, mu_c=mu_c, seed=seed)
  unordered_measurements = [m for mt in measurements for m in mt]

  # Initialize tracks
  tm = ConstantVelocity(ndim_pos=2, q=0.005)
  mm = LinearMeasurementModel(
      ndim_state=4,
      covar=np.diag([0.75, 0.75]),
      measured_dims=[0, 2],
  )
  tracks = []
  for path in paths:
    init_state = path[0]
    init_state.covar = np.diag([1.5, 0.5, 1.5, 0.5])
    tracks.append(Track(history=init_state))

  # Perform track filtering + joint association
  spa = TotalSPA()
  kf = KalmanFilter(
      transition_model=tm,
      measurement_model=mm,
  )
  ukf = UnscentedKalmanFilter(
      transition_model=tm,
      measurement_model=mm,
  )
  current_time = last_update = 0
  for iglobal, current_measurements in enumerate(measurements):
    if len(current_measurements) == 0:
      continue

    current_time = current_measurements[0].timestamp
    ms = [m.measurement for m in current_measurements]
    dt = current_time - last_update

    # Gaussian mixture components
    Mn, Nt = len(ms), len(tracks)
    nx, nz = tm.ndim_state, mm.ndim_meas
    
    x_preds = []
    P_preds = []
    x = np.empty((Mn+1, Nt, nx))
    P = np.empty((Mn+1, Nt, nx, nx))
    z_pred = np.empty((Nt, nz))
    S = np.empty((Nt, nz, nz))
    for it in range(Nt):
      x_pred, P_pred = ukf.predict(
          dt=dt, state=tracks[it].state, covar=tracks[it].covar)
      x_preds.append(x_pred), P_preds.append(P_pred)
      for im in range(Mn):
        x[im, it], P[im, it], S[it], _, z_pred[it] = ukf.update(
            measurement=ms[im],
            predicted_state=x_preds[it],
            predicted_covar=P_preds[it])
    x[-1] = np.array(x_preds)
    P[-1] = np.array(P_preds)
    # Gaussian mixture weights
    lz = np.empty((Nt, Mn))
    for it in range(Nt):
      lz[it] = multivariate_normal.pdf(ms, mean=z_pred[it], cov=S[it])
    # TODO: This should be proportional to a target-specific gate volume
    Vc = (10 + 10) * (10 + 10)
    lam_c = mu_c / Vc
    w = list(spa.weights(pd=pd, lz=lz, lam_c=lam_c, niter=10))
    # Perform gaussian mixture with the weights and parameters above
    x_post = np.empty_like(x_preds)
    P_post = np.zeros_like(P_preds)
    for it in range(Nt):
      x_post[it], P_post[it] = mix_gaussians(means=list(x[:, it, :]),
                                           covars=list(P[:, it]),
                                           weights=w[it])
      tracks[it].append(state=State(
          state=x_post[it], covar=P_post[it], timestamp=current_time))

    last_update = current_time

  true_states = np.array([[state.state for state in path] for path in paths])
  all_measurements = np.array(
      [m.measurement for m in unordered_measurements])
  t1_pos = np.array([state.state[[0, 2]] for state in tracks[0]])
  plt.figure()
  # Plot states
  plt.plot(true_states[0, :, 0], true_states[0, :, 2], '--', label='Target 1')
  # Plot measurements
  plt.plot(all_measurements[:, 0],
           all_measurements[:, 1], 'o', label='Measurements')
  plt.plot(t1_pos[:, 0], t1_pos[:, 1], label='Track 1')
  t2_pos = np.array([state.state[[0, 2]] for state in tracks[1]])
  plt.plot(true_states[1, :, 0], true_states[1, :, 2], '--', label='Target 2')
  plt.plot(t2_pos[:, 0], t2_pos[:, 1], label='Track 2')
  plt.grid()
  plt.legend()
  plt.show()


def test_spada():
  pass


if __name__ == '__main__':
  test_spa()
  # test_spada()
