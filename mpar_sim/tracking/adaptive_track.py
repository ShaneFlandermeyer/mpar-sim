from functools import lru_cache
from typing import Callable, List
import numpy as np

from mpar_sim.common.coordinate_transform import cart2sph_covar
from mpar_sim.models.transition.base import TransitionModel
from mpar_sim.models.transition.linear import ConstantVelocity
from mpar_sim.tracking.kalman import kalman_predict


def adaptive_revisit_interval(state_vector: np.ndarray,
                              covar: np.ndarray,
                              predict_func: Callable,
                              transition_model: TransitionModel,
                              beamwidths: np.ndarray,
                              track_sharpness: float = 0.05,
                              min_revisit_interval: float = 0.2,
                              max_revisit_interval: float = 2.0,
                              position_mapping: List[int] = [0, 2, 4],
                              ) -> float:
  """
  Compute the maximum revisit time for the track based on its covariance.
  """
  # Compute an array of possible revisit times to consider
  tmin = min_revisit_interval
  tmax = max_revisit_interval
  n = int(np.ceil(np.log(tmax/tmin)/np.log(2)))
  revisit_times = tmin * np.power(2, np.arange(n-1))
  revisit_times = np.append(revisit_times, tmax)

  for dt in reversed(revisit_times):
    predicted_state, predicted_covar = predict_func(state=state_vector,
                                                    covar=covar,
                                                    transition_model=transition_model,
                                                    time_interval=dt)

    # Convert the covariance matrix from Cartesian to spherical coordinates
    position_xyz = predicted_state[position_mapping].ravel()
    position_covar_xyz = predicted_covar[position_mapping,
                                         :][:, position_mapping]
    position_covar_sph = cart2sph_covar(position_covar_xyz, *position_xyz)

    # Compute the error of the track in az/el,and determine the revisit interval from the track sharpness
    error_std_dev = np.sqrt(np.diagonal(position_covar_sph))
    az_error, el_error, range_error = error_std_dev
    az_error = np.rad2deg(az_error)
    el_error = np.rad2deg(el_error)

    az_threshold = track_sharpness * beamwidths[0]
    el_threshold = track_sharpness * beamwidths[1]
    if az_error < az_threshold and el_error < el_threshold:
      return dt

  # If the track error is never within the limits, return the minimum revisit interval
  return dt