from mpar_sim.predictor.kalman import kalman_predict
from mpar_sim.updater.kalman import kalman_update, extended_kalman_update
import numpy as np
from filterpy.kalman import predict, update


class TestLinearTransitionModel():
  def function(self, prior, dt, noise=False):
    if noise:
      process_noise = self.covar()
    else:
      process_noise = 0

    return self.matrix(dt) @ prior + process_noise

  def matrix(self, dt):
    return np.array([[1, dt], [0, 1]])

  def covar(self):
    return np.array([[0.588, 1.175],
                     [1.175, 2.35]])


class TestLinearMeasurementModel():
  def function(self, state):
    # TODO: No measurement noise
    return self.matrix() @ state

  def matrix(self):
    return np.atleast_2d(np.array([1, 0]))

  def covar(self):
    return np.array([[5.]])


def test_kalman_predict():
  transition_model = TestLinearTransitionModel()
  x = np.array([11.35, 4.5])
  P = np.array([[545, 150], [150, 500]])
  dt = 0.3
  F = transition_model.matrix(dt)
  Q = transition_model.covar()
  x_actual, P_actual = kalman_predict(prior_state=x,
                                      prior_covar=P,
                                      transition_matrix=F,
                                      noise_covar=Q)
  x_expected, P_expected = predict(x=x, P=P, F=F, Q=Q)
  if np.all(abs(x_actual - x_expected) < 1e-10) and np.all(abs(P_actual - P_expected) < 1e-10):
    print("- test_kalman_predict: PASSED")
  else:
    print("- test_kalman_predict: FAILED")


def test_kalman_update():
  measurement_model = TestLinearMeasurementModel()
  z = 1
  x = np.array([12.7, 4.5])
  P = np.array([[680.587, 301.175],
                [301.175, 502.35]])
  R = measurement_model.covar()
  H = measurement_model.matrix()
  x_actual, P_actual = kalman_update(
      prior_state=x,
      prior_covar=P,
      measurement=z,
      measurement_model=measurement_model,
  )
  x_expected, P_expected = update(x=x, P=P, z=z, R=R, H=H)
  if np.all(abs(x_actual - x_expected) < 1e-10) and np.all(abs(P_actual - P_expected) < 1e-10):
    print("- test_kalman_update: PASSED")
  else:
    print("- test_kalman_update: FAILED")

def test_extended_kalman_update():
  print("- test_extended_kalman_update: NOT IMPLEMENTED")
  

if __name__ == '__main__':
  test_kalman_predict()
  test_kalman_update()
  test_extended_kalman_update()
