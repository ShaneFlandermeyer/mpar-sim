#
# Author: Shane Flandermeyer
# Created on Thu Feb 16 2023
# Copyright (c) 2023
#
# This file provides functions for adaptive tracking algorithms.
#


import datetime
from typing import Callable, List, Tuple, Union

import numpy as np

from mpar_sim.common.coordinate_transform import cart2sph_covar
from mpar_sim.models.transition.base import TransitionModel
from mpar_sim.types.detection import Detection
from mpar_sim.types.look import Look
from mpar_sim.types.track import Track


class AdaptiveTrackManager():
  def __init__(self,
               track_sharpness: float = 0.15,
               confirmation_interval: float = 0.05,
               min_revisit_interval: float = 0.2,
               max_revisit_interval: float = 2.0,
               position_mapping: List[int] = [0, 2, 4],
               velocity_mapping: List[int] = [1, 3, 5],
               ):
    # TODO: Add a tracker object. This object should be used for:
    #   - Prediction
    #   - Update
    #   - Transition/measurement models
    #   - Initiation logic
    self.track_sharpness = track_sharpness
    self.confirmation_interval = confirmation_interval
    self.min_revisit_interval = min_revisit_interval
    self.max_revisit_interval = max_revisit_interval
    self.position_mapping = position_mapping
    self.velocity_mapping = velocity_mapping

    self.confirmed_tracks = []
    self.tentative_tracks = []
    self.update_times = {}

  def process_detections(self,
                         detections: List[Detection],
                         time: Union[float, datetime.datetime],
                         # TODO: Need this for the adaptive revisit interval
                         look: Look = None,
                         ) -> None:
    for detection in detections:
      # Check if the target is already in the list of tracks.
      target_id = detection.groundtruth_path.id
      confirmed_target_ids = [
          track.target_id for track in self.confirmed_tracks]
      tentative_target_ids = [
          track.target_id for track in self.tentative_tracks]
      if target_id in confirmed_target_ids:
        track = self.confirmed_tracks[confirmed_target_ids.index(target_id)]
        # TODO: Make this work properly
        # dt = adaptive_revisit_interval(track.state_vector,
        #                                track.covar,
        #                                self.tracker.predict,
        #                                self.tracker.transition_model,
        #                                look.az_beamwidth,
        #                                look.el_beamwidth,
        #                                self.track_sharpness,
        #                                self.min_revisit_interval,
        #                                self.max_revisit_interval,
        #                                self.position_mapping,
        #                                )
        dt = 2.0
      elif target_id in tentative_target_ids:
        track = self.tentative_tracks[tentative_target_ids.index(target_id)]
        dt = self.confirmation_interval
      else:
        track = Track(target_id=target_id)
        dt = self.confirmation_interval
        self.tentative_tracks.append(track)

      if isinstance(time, datetime.datetime):
        dt = datetime.timedelta(seconds=dt)
      self.update_times[track.id] = time + dt

  # def generate_look(self, time: Union[float, datetime.datetime]) -> Look:
    # pass


def adaptive_revisit_interval(state_vector: np.ndarray,
                              covar: np.ndarray,
                              predict_func: Callable,
                              transition_model: TransitionModel,
                              az_beamwidth: float,
                              el_beamwidth: float,
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

    az_threshold = track_sharpness * az_beamwidth
    el_threshold = track_sharpness * el_beamwidth
    if az_error < az_threshold and el_error < el_threshold:
      return dt

  # If the track error is never within the limits, return the minimum revisit interval
  return dt


if __name__ == '__main__':
  t1 = Track()
  t2 = Track()
