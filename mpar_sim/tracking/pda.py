from typing import List, Tuple
import numpy as np
from mpar_sim.models.transition import ConstantVelocity
from mpar_sim.models.measurement.linear import LinearMeasurementModel
import matplotlib.pyplot as plt
from mpar_sim.tracking.kalman import KalmanFilter
from mpar_sim.types.detection import FalseDetection, TrueDetection

from mpar_sim.types.trajectory import Trajectory
import numpy as np
from scipy.stats import multivariate_normal
from mpar_sim.tracking.gate import gate_volume, gate_threshold, ellipsoid_gate
from scipy.stats import uniform


class PDAFilter():
  def __init__(self,
               state_filter: KalmanFilter,
               pd: float = 0.90,
               pg: float = 0.99,
               clutter_density: float = None,
               ):
    self.filter = state_filter
    self.pd = pd
    self.pg = pg
    self.clutter_density = clutter_density
    
  def gate(self, measurements: List[np.ndarray]) -> np.ndarray:
    G = gate_threshold(pg=self.pg,
                       ndim=self.measurement_model.ndim)
    in_gate = ellipsoid_gate(measurements=measurements,
                             predicted_measurement=self.filter.z_pred,
                             innovation_covar=self.filter.S,
                             threshold=G)
    return [m for m, g in zip(measurements, in_gate) if g]

  def update(self,
             measurements: List[np.ndarray],
             dt: float,
             ) -> Tuple[np.ndarray]:
    # Get the predicted state/covariance/measurement, along with the innovation covariance and Kalman gain.
    self.filter.predict(dt)
    self.filter.update(measurement=np.empty(self.measurement_model.ndim))

    # Handle gating
    gated_measurements = self.gate(measurements)
    # Compute clutter density.
    if self.clutter_density:
      clutter_density = self.clutter_density
    else:
      m = len(gated_measurements)
      # For m validated measurements, the clutter density is m / V
      V_gate = gate_volume(innovation_covar=self.filter.S,
                           gate_probability=self.pg,
                           ndim=self.measurement_model.ndim)
      clutter_density = m / V_gate

    # Compute association probabilities for gated measurements
    if len(gated_measurements) == 0:
      betas = np.ones(1,)
    else:
      betas = self._association_probs(
          z=gated_measurements,
          z_pred=self.filter.z_pred,
          S=self.filter.S,
          pd=self.pd,
          pg=self.pg,
          clutter_density=clutter_density,
      )

    self.filter.x, self.filter.P = self._update_state(
        z=gated_measurements,
        x_pred=self.filter.x_pred,
        P_pred=self.filter.P_pred,
        K=self.filter.K,
        z_pred=self.filter.z_pred,
        S=self.filter.S,
        betas=betas,
    )
    return self.filter.x, self.filter.P

  @staticmethod
  def _association_probs(
      z: List[np.array],
      z_pred: np.array,
      S: np.array,
      pd: float,
      pg: float,
      clutter_density: float,
  ) -> np.ndarray:
    """
    Compute the association probabilities for each measurement in the list of gated measurements.

    Parameters
    ----------
    z : List[np.array]
        Gated measurements
    z_pred : np.array
        Predicted track measurement
    S : np.array
        Innovation covar
    pd : float
        Probability of detection
    pg : float
        Gate probability
    clutter_density : float
        Density of the spatial Poisson process that models the clutter

    Returns
    -------
    np.ndarray
        Length-m+1 array of association probabilities. The first element is the probability of no detection.
    """
    m = len(z)
    betas = np.empty(m+1)
    # Probability of no detection
    betas[0] = 1 - pd*pg
    # Probability of each detection from likelihood ratio
    l = multivariate_normal.pdf(
        z,
        mean=z_pred,
        cov=S,
    )
    l_ratio = l * pd / clutter_density
    betas[1:] = l_ratio

    # Normalize to sum to 1
    betas /= np.sum(betas)
    return betas

  @staticmethod
  def _update_state(
      z: List[np.array],
      # Filter parameters
      x_pred: np.array,
      P_pred: np.array,
      z_pred: np.array,
      K: np.array,
      S: np.array,
      betas: np.ndarray,
  ) -> Tuple[np.ndarray]:
    """
    Compute the posterior state and covariance for the given track as a Gaussian mixture

    Parameters
    ----------
    z : List[np.array]
      List of gated measurements
    x_pred : np.array
        Predicted state
    P_pred : np.array
        Predicted covar
    z_pred : np.array
        Predicted measurement
    K : np.array
        Kalman gain matrix
    S : np.array
        Innovation covar
    betas : np.ndarray
        Association probabilities

    Returns
    -------
    Tuple[np.ndarray]
        Posterior state and covar
    """
    # If there are no gated measurements, return the predicted state and covariance
    if len(z) == 0:
      return x_pred, P_pred

    # State estimation
    # Bar-Shalom2009 - Equations 39-40
    y = np.array(z) - z_pred
    v = np.einsum('m, mi->i', betas[1:], y)
    x_post = x_pred + K @ v

    # Bar-Shalom2009 - Equations 42-44
    betaz = np.einsum('m, mi->mi', betas[1:], y)
    S_mix = np.einsum('mi, mj->ij', betaz, y) - np.outer(v, v)
    Pc = P_pred - K @ S @ K.T
    Pt = K @ S_mix @ K.T
    P_post = betas[0]*P_pred + (1 - betas[0])*Pc + Pt

    return x_post, P_post
  
  def predict(self, dt: float):
    self.filter.predict(dt)

  @property
  def x(self):
    return self.filter.x

  @property
  def P(self):
    return self.filter.P

  @property
  def transition_model(self):
    return self.filter.transition_model

  @property
  def measurement_model(self):
    return self.filter.measurement_model


def test_pda():
  # Generate ground truth
  current_time = last_update = 0
  dt = 1
  seed = 0
  np.random.seed(seed)
  transition_model = ConstantVelocity(ndim_pos=2,
                                      noise_diff_coeff=0.005,
                                      seed=seed)
  trajectory = Trajectory()
  trajectory.append(state=np.array([0, 1, 0, 1]),
                    covar=np.diag([1.5, 0.5, 1.5, 0.5]),
                    timestamp=current_time)
  trajectory.step(transition_model, dt=dt, nsteps=100, noise=True)

  # Generate measurements
  measurement_model = LinearMeasurementModel(
      ndim_state=4,
      covar=np.array([[0.75, 0], [0, 0.75]]),
      measured_dims=[0, 2],
      seed=seed,
  )
  pd = 0.9
  kf = KalmanFilter(
      x=trajectory[0].state,
      P=trajectory[0].covar,
      transition_model=transition_model,
      measurement_model=measurement_model,
  )
  pda = PDAFilter(
      state_filter=kf,
      pd=pd,
      pg=0.95,
      # clutter_density=0.125,
  )
  detections = []
  track_pos = []
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

    # Gate detections
    if len(current_detections) > 0:
      pda.update(measurements=[
                 d.measurement for d in current_detections], dt=current_time - last_update)
      track_pos.append(pda.x[np.array([0, 2])])
      detections.extend(current_detections)
      last_update = current_time
    current_time += dt

  states = np.array([state.state for state in trajectory])
  true_detections = np.array(
      [detection.measurement for detection in detections if isinstance(detection, TrueDetection)])
  track_pos = np.array(track_pos)
  false_detections = np.array(
      [detection.measurement for detection in detections if isinstance(detection, FalseDetection)])
  plt.plot(states[:, 0], states[:, 2], label='Ground Truth')
  plt.plot(true_detections[:, 0], true_detections[:, 1],
           'o', label='True Detections')
  plt.plot(false_detections[:, 0], false_detections[:,
           1], 'x', label='False Detections')
  plt.plot(track_pos[:, 0], track_pos[:, 1], '.-', label='Track')
  # plt.xlim([-40, 60])
  # plt.ylim([-10, 60])
  plt.show()


if __name__ == '__main__':
  test_pda()