import matplotlib.pyplot as plt
import numpy as np
import pytest

from mpar_sim.models.measurement import LinearMeasurementModel
from mpar_sim.models.transition import ConstantVelocity
from mpar_sim.tracking import JPDATracker, KalmanFilter
from mpar_sim.types import FalseDetection, Track, Trajectory, TrueDetection
from mpar_sim.types.state import State
from scipy.stats import multivariate_normal


def generate_trajectories(seed=None):
  # Test SPA methods on a simple crossing target scenario
  n_steps = 50
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


def spada(Mn):
  # TODO: Implement
  # kappa = 1 should correspond to standard PDA
  return np.ones(Mn+1)


def test_spa():
  # Parameters
  pd = 0.9
  mu_c = 5

  seed = 0
  np.random.seed(seed)

  paths = generate_trajectories(seed=seed)
  paths = [paths[0]]
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
  current_time = last_update = 0
  for i, current_measurements in enumerate(measurements):
    if len(current_measurements) == 0:
      continue
      
    current_time = current_measurements[0].timestamp
    ms = [m.measurement for m in current_measurements]
    dt = current_time - last_update

    # SPA stuff
    # Prediction step
    F = tm.matrix(dt)
    Q = tm.covar(dt)
    x_prior = tracks[0].state
    P_prior = tracks[0].covar
    x_pred = F @ x_prior
    P_pred = F @ P_prior @ F.T + Q
    # Gaussian mixture components
    H = mm.matrix()
    R = mm.covar()
    K = P_pred @ H.T @ np.linalg.inv(H @ P_pred @ H.T + R)
    Mn = len(ms)
    x, P = np.empty((Mn+1, x_prior.size)), np.empty((Mn+1, *P_prior.shape))
    for m in range(len(ms)):
      x[m] = x_pred + K @ (ms[m] - H @ x_pred)
      P[m] = P_pred - K @ H @ P_pred
    x[-1] = x_pred
    P[-1] = P_pred
    # TODO: Data association probabilities
    kappa = spada(Mn=Mn)
    # Gaussian mixture weights
    lz = multivariate_normal.pdf(ms, mean=H@x_pred, cov=H@P_pred@H.T + R)
    Vc = (10 + 10) * (10 + 10)
    w = np.empty((Mn+1))
    w[:-1] = pd * lz * kappa[1:] / (mu_c / Vc)
    w[-1] = (1 - pd) * kappa[0]
    w /= np.sum(w)
    # Perform gaussian mixture with the weights and parameters above
    x_post = np.dot(w, x)
    # TODO: Do without a loop
    P_post = np.zeros_like(P[0])
    for m in range(Mn+1):
      P_post += w[m] * (P[m] + x[m] @ x[m].T)
    P_post -= x_post @ x_post.T

    last_update = current_time
    tracks[0].append(state=State(
        state=x_post, covar=P_post, timestamp=current_time))

  true_states = np.array([[state.state for state in path] for path in paths])
  all_measurements = np.array(
      [m.measurement for m in unordered_measurements])
  track_pos = np.array([state.state[[0, 2]] for state in tracks[0]])
  plt.figure()
  # Plot states
  plt.plot(true_states[0, :, 0], true_states[0, :, 2], '--', label='Target 1')
  # plt.plot(true_states[1, :, 0], true_states[1, :, 2], '--', label='Target 2')
  # Plot measurements
  plt.plot(all_measurements[:, 0],
           all_measurements[:, 1], 'o', label='Measurements')
  plt.plot(track_pos[:, 0], track_pos[:, 1], label='Track')
  plt.grid()
  plt.legend()
  plt.show()


if __name__ == '__main__':
  test_spa()
