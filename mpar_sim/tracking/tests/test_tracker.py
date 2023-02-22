import datetime

import numpy as np
import pytest
from filterpy.kalman import predict, update

from mpar_sim.tracking.kalman import kalman_predict, kalman_update
from mpar_sim.tracking.tracker import Tracker
from mpar_sim.types.detection import Detection
from mpar_sim.types.state import State
from mpar_sim.types.track import Track
from mpar_sim.models.measurement.nonlinear import CartesianToRangeAzElRangeRate


class TestLinearMeasurementModel():
  def function(self, state):
    return self.matrix() @ state

  def matrix(self):
    return np.atleast_2d(np.array([1, 0]))

  def covar(self, dt=None):
    return np.array([[5.]])


class TestLinearTransitionModel():
  def function(self, prior, dt, noise=False):
    if noise:
      process_noise = self.covar()
    else:
      process_noise = 0

    return self.matrix(dt) @ prior + process_noise

  def matrix(self, dt):
    return np.array([[1, dt], [0, 1]])

  def covar(self, dt=None):
    return np.array([[0.588, 1.175],
                     [1.175, 2.35]])


def test_predict():
  tracker = Tracker(
      predict_func=kalman_predict,
      transition_model=TestLinearTransitionModel(),
      update_func=None,
      measurement_model=None,
  )

  # Add a new track update
  # This example comes from the kalman filter ebook
  track = Track()
  state = State(
      state_vector=np.array([11.35, 4.5]),
      covar=np.array([[545, 150], [150, 500]]),
      timestamp=0.0,
  )
  track.append(state)

  # Predict the track
  dt = 0.3
  time = track.timestamp + dt
  x_actual, P_actual = tracker.predict(track, time)

  # Compare to KF ebook object
  F = tracker.transition_model.matrix(dt)
  Q = tracker.transition_model.covar()
  x_expected, P_expected = predict(
      x=track.state_vector, P=track.covar, F=F, Q=Q)
  assert np.allclose(x_actual, x_expected)
  assert np.allclose(P_actual, P_expected)


def test_update():
  tracker = Tracker(
      update_func=kalman_update,
      measurement_model=TestLinearMeasurementModel(),
      predict_func=None,
      transition_model=None,
  )

  # Add a new track update
  # This example comes from the kalman filter ebook
  track = Track()
  state = State(
      state_vector=np.array([12.7, 4.5]),
      covar=np.array([[680.587, 301.175],
                      [301.175, 502.35]]),
  )
  track.append(state)
  measurement = 1
  x_actual, P_actual = tracker.update(track, measurement)

  # Compare to KF ebook object
  R = tracker.measurement_model.covar()
  H = tracker.measurement_model.matrix()
  x_expected, P_expected = update(
      x=track.state_vector, P=track.covar, z=measurement, R=R, H=H)

  assert np.allclose(x_actual, x_expected)
  assert np.allclose(P_actual, P_expected)


def test_initiate():
  mm = CartesianToRangeAzElRangeRate(
    noise_covar=np.diag([3, 2.5, 2, 1])**2,
    translation_offset=np.array([25, -40, 0]),
    velocity=np.array([0, 5, 0])
  )
  tracker = Tracker(
      update_func=kalman_update,
      measurement_model=mm,
      predict_func=None,
      transition_model=None,
  )
  detection = Detection(state_vector=np.array([45, -10, 1000, -4]))
  tracker.initiate(detection)
  


if __name__ == '__main__':
  pytest.main()
  # test_initiate()
