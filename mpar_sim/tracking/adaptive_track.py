from functools import lru_cache
from typing import Callable, List
import numpy as np

from mpar_sim.common.coordinate_transform import cart2sph_covar


# class AdaptiveTracker():
#   def __init__(self,
#                # Tracker components
#                predict_func: Callable,
#                # Adaptive track parameters
#                track_sharpness: float = 0.05,
#                min_revisit_interval: float = 0.2,
#                max_revisit_interval: float = 2.0,
#                position_mapping: List[int] = [0, 2, 4],
#                ):
#     self.predict_func = predict_func
#     self.track_sharpness = track_sharpness
#     self.min_revisit_interval = min_revisit_interval
#     self.max_revisit_interval = max_revisit_interval
#     self.position_mapping = position_mapping

#     # Compute array of possible revisit times
#     tmin = min_revisit_interval
#     tmax = max_revisit_interval
#     n = int(np.ceil(np.log(tmax/tmin)/np.log(2)))
#     self.revisit_times = tmin * np.power(2, np.arange(n-1))
#     self.revisit_times = np.append(self.revisit_times, tmax)

#     for dt in reversed(self.revisit_times):
#       # TODO: May need to re-work the Kalman filter functions for this.
#       predicted_state = self.predict_func(state_vector, covar)

#       # Convert the covariance matrix from Cartesian to spherical coordinates
#       position_xyz = predicted_state[self.position_mapping].ravel()
#       position_covar_xyz = covar[self.position_mapping,
#                                  :][:, self.position_mapping]
#       position_covar_sph = cart2sph_covar(position_covar_xyz, *position_xyz)

#       # Compute the error of the track in az/el,and determine the revisit interval from the track sharpness
#       error_std_dev = np.sqrt(np.diagonal(position_covar_sph))
#       az_error, el_error, range_error = error_std_dev
#       az_threshold = self.track_sharpness * np.deg2rad(self.azimuth_beamwidth)
#       el_threshold = self.track_sharpness * \
#           np.deg2rad(self.elevation_beamwidth)
#       if az_error < az_threshold and el_error < el_threshold:
#         return dt

#   # @lru_cache
#   def compute_revisit_time(self, state_vector: np.ndarray, covar: np.ndarray) -> float:
#     """
#     Compute the maximum revisit time for the track.
#     TODO: Consider making a track object for this. Numpy array is fine for now
#     """
#     for dt in reversed(self.revisit_times):
#       # TODO: May need to re-work the Kalman filter functions for this.
#       predicted_state = self.predict_func(state_vector, covar)

#       # Convert the covariance matrix from Cartesian to spherical coordinates
#       position_xyz = predicted_state[self.position_mapping].ravel()
#       position_covar_xyz = covar[self.position_mapping,
#                                  :][:, self.position_mapping]
#       position_covar_sph = cart2sph_covar(position_covar_xyz, *position_xyz)

#       # Compute the error of the track in az/el,and determine the revisit interval from the track sharpness
#       error_std_dev = np.sqrt(np.diagonal(position_covar_sph))
#       az_error, el_error, range_error = error_std_dev
#       az_threshold = self.track_sharpness * np.deg2rad(self.azimuth_beamwidth)
#       el_threshold = self.track_sharpness * \
#           np.deg2rad(self.elevation_beamwidth)
#       if az_error < az_threshold and el_error < el_threshold:
#         return dt

#     # If the track error is never within the limits, return the minimum revisit interval
#     return dt

def adaptive_revisit_interval(state_vector: np.ndarray,
                              covar: np.ndarray,
                              predict_func: Callable,
                              transition_func: Callable,
                              beamwidths: np.ndarray,
                              track_sharpness: float = 0.05,
                              min_revisit_interval: float = 0.2,
                              max_revisit_interval: float = 2.0,
                              position_mapping: List[int] = [0, 2, 4],
                              ) -> float:
  """
  Compute the maximum revisit time for the track.
  """
  tmin = min_revisit_interval
  tmax = max_revisit_interval
  n = int(np.ceil(np.log(tmax/tmin)/np.log(2)))
  revisit_times = tmin * np.power(2, np.arange(n-1))
  revisit_times = np.append(revisit_times, tmax)

  for dt in reversed(revisit_times):
    # TODO: May need to re-work the Kalman filter functions for this.
    predicted_state = predict_func(state_vector, covar, dt)

    # Convert the covariance matrix from Cartesian to spherical coordinates
    position_xyz = predicted_state[position_mapping].ravel()
    position_covar_xyz = covar[position_mapping, :][:, position_mapping]
    position_covar_sph = cart2sph_covar(position_covar_xyz, *position_xyz)

    # Compute the error of the track in az/el,and determine the revisit interval from the track sharpness
    error_std_dev = np.sqrt(np.diagonal(position_covar_sph))
    az_error, el_error, range_error = error_std_dev
    az_threshold = track_sharpness * np.deg2rad(beamwidths[0])
    el_threshold = track_sharpness * np.deg2rad(beamwidths[1])
    if az_error < az_threshold and el_error < el_threshold:
      return dt

  # If the track error is never within the limits, return the minimum revisit interval
  return dt


if __name__ == '__main__':
  state_vector = np.random.uniform(low=0, high=100, size=(6, 1))
  covar = np.diag(np.random.uniform(low=0, high=0.1, size=6))
  transition_matrix = np.diag(np.ones((6,)))
  def predict_func(x, P, dt): return transition_matrix @ x
  track_sharpness = 0.10
  min_revisit_interval = 0.2
  max_revisit_interval = 2.0

  # tracker = AdaptiveTracker(predict_func=lambda x, P: transition_matrix @ x)
  dt = adaptive_revisit_interval(state_vector,
                                 covar,
                                 predict_func=predict_func,
                                 beamwidths=[2, 2],
                                 track_sharpness=track_sharpness,
                                 min_revisit_interval=min_revisit_interval,
                                 max_revisit_interval=max_revisit_interval,)
  print(dt)
