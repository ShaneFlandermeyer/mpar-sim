import numpy as np
from filterpy.common import kinematic_kf
import filterpy.kalman
from mpar_sim.tracking import IMMEstimator, KalmanFilter
import pytest


class DummyTransition():
  def __init__(self, filter):
    self.filter = filter

  def __call__(self, x):
    return self.matrix() @ x

  def matrix(self, dt):
    return self.filter.F

  def covar(self, dt):
    return self.filter.Q


class DummyMeasurement():
  def __init__(self, filter):
    self.filter = filter

  def __call__(self, x):
    return self.matrix() @ x

  def matrix(self):
    return self.filter.H

  def covar(self):
    return self.filter.R

def test_imm():
  # Compare the results of my IMM implementation to filterpy
  # Create two KFs
  kf1 = kinematic_kf(2, 2)
  kf2 = kinematic_kf(2, 2)
  # do some settings of x, R, P etc. here, I'll just use the defaults
  kf2.Q *= 0   # no prediction error in second filter

  # Create mpar-sim variants
  my_kf1 = KalmanFilter(transition_model=DummyTransition(kf1),
                        measurement_model=DummyMeasurement(kf1))
  my_kf2 = KalmanFilter(transition_model=DummyTransition(kf2),
                        measurement_model=DummyMeasurement(kf2))
  my_imm = IMMEstimator(filters=[my_kf1, my_kf2],
                        init_probs=[0.5, 0.5],
                        transition_probs=np.array([[0.97, 0.03], [0.03, 0.97]]))

  filters = [kf1, kf2]
  mu = [0.5, 0.5]  # each filter is equally likely at the start
  trans = np.array([[0.97, 0.03], [0.03, 0.97]])
  imm = filterpy.kalman.IMMEstimator(filters, mu, trans)

  states = [kf1.x.squeeze(), kf2.x.squeeze()]
  covars = [kf1.P, kf2.P]
  for i in range(100):
    # make some noisy data
    x = i + np.random.randn()*np.sqrt(kf1.R[0, 0])
    y = i + np.random.randn()*np.sqrt(kf1.R[1, 1])
    z = np.array([x, y])

    # perform predict/update cycle for each filter
    pred_states, pred_covars = my_imm.predict(
        states=states, covars=covars, dt=1)
    states, covars, x, P = my_imm.update(
      measurement=z,
      predicted_states=pred_states,
      predicted_covars=pred_covars,
    )
    imm.predict()
    imm.update(z)
    
    assert np.allclose(my_imm.mode_probs, imm.mu)
    assert np.allclose(x, imm.x.squeeze())
    assert np.allclose(P, imm.P.squeeze())

if __name__ == '__main__':
  pytest.main()