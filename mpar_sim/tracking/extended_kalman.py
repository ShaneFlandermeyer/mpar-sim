import datetime
from typing import Tuple, Union
import numpy as np

from mpar_sim.models.measurement.base import MeasurementModel
from mpar_sim.models.transition.base import TransitionModel


def extended_kalman_predict(state: np.ndarray,
                            covar: np.ndarray,
                            transition_model: TransitionModel,
                            time_interval: Union[float, datetime.timedelta]
                            ) -> Tuple[np.ndarray, np.ndarray]:
  """
  Extended Kalman predict step.  

  Equations: https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python/blob/master/11-Extended-Kalman-Filters.ipynb  

  Parameters
  ----------
  state : np.ndarray
      Prior state vector
  covar : np.ndarray
      Prior covariance matrix
  transition_model : TransitionModel
      Transition model object. This object must define the following methods:
          - function(state, dt): Propagates the state vector forward by dt
          - jacobian(dt): Returns the jacobian of the transition function
          - covar(dt): Returns the transition noise covariance matrix
  time_interval : Union[float, datetime.timedelta]
      Time interval over which to predict  

  Returns
  -------
  Tuple[np.ndarray, np.ndarray]
       - Predicted state vector
       - Predicted covariance
  """
  transition_matrix = transition_model.jacobian(time_interval)
  noise_covar = transition_model.covar(time_interval)

  # Propagate the state forward in time
  predicted_state = transition_model.function(state, time_interval)
  predicted_covar = transition_matrix @ covar @ transition_matrix.T + noise_covar

  return predicted_state, predicted_covar


def extended_kalman_update(state: np.ndarray,
                           covar: np.ndarray,
                           measurement: np.ndarray,
                           measurement_model: MeasurementModel,
                           ) -> Tuple[np.ndarray, np.ndarray]:
  """
  Perform the extended Kalman update step. Unlike the traditional Kalman filter, the extended Kalman filter works for nonlinear measurement models. 
  The EKF is almost identical to the KF, but the nonlinear measurement must be linearized by using the Jacobian for the measurement matrix H.

  See equations here:
  https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python/blob/master/11-Extended-Kalman-Filters.ipynb

  The posterior covariance is computed using the equation from here:
  https://stonesoup.readthedocs.io/en/v0.1b5/stonesoup.updater.html?highlight=kalman#module-stonesoup.updater.kalman

  Parameters
  ----------
  predicted_state : np.ndarray
      Predicted state vector (mean)
  predicted_covar : np.ndarray
      Predicted covariance
  measurement : np.ndarray
      Actual measurement
  measurement_model : MeasurementModel
      Model used to collect measurement. This object must define the following methods:
        - function(state): Convert the input from state space into measurement space 
        - jacobian(): Returns the jacobian of the measurement function
        - covar(): Returns the measurement noise covariance matrix

  Returns
  -------
  Tuple[np.ndarray, np.ndarray]
      - Predicted state vector
      - Predicted covariance
  """
  state = state.ravel()
  measurement = measurement.ravel()
  # Compute the residual
  prior_measurement = measurement_model(state, noise=False)
  # TODO: This does not handle aliasing of angles properly
  residual = measurement - prior_measurement.ravel()

  # Compute the Kalman gain
  measurement_matrix = measurement_model.jacobian(state)
  noise_covar = measurement_model.covar()
  measurement_cross_covar = covar @ measurement_matrix.T
  innovation_covar = measurement_matrix @ measurement_cross_covar + noise_covar
  kalman_gain = measurement_cross_covar @ np.linalg.inv(innovation_covar)

  # Compute the updated state and covariance
  posterior_mean = state + kalman_gain @ residual
  posterior_covar = covar - kalman_gain @ innovation_covar @ kalman_gain.T

  return posterior_mean, posterior_covar
