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

from mpar_sim.common.coordinate_transform import cart2sph, cart2sph_covar
from mpar_sim.models.transition.base import TransitionModel
from mpar_sim.tracking.tracker import Tracker
from mpar_sim.types.detection import Detection
from mpar_sim.types.look import Look, TrackConfirmationLook, TrackUpdateLook
from mpar_sim.types.state import State
from mpar_sim.types.track import Track


class AdaptiveTrackManager():
  def __init__(self,
               tracker: Tracker,
               track_sharpness: float = 0.15,
               confirmation_interval: float = 0.05,
               min_revisit_interval: float = 0.2,
               max_revisit_interval: float = 2.0,
               position_mapping: List[int] = [0, 2, 4],
               velocity_mapping: List[int] = [1, 3, 5],
               n_confirm_detections: int = 3,
               ):
    """
    This class implements an adaptive track manager that selects the revisit interval for each track based on the angular error from the track covariance matrix.

    The implementation is based on section 3.2.3 of [1].

    Parameters
    ----------
    tracker : Tracker
        An object that implements a tracking algorithm. The following methods are required:
          - predict(state, time)
          - update(state, measurement)
          - initiate(measurement)
    track_sharpness : float, optional
        The angular error threshold (in fractions of a beamwidth). Track updates are scheduled such that this is not reached, by default 0.15
    confirmation_interval : float, optional
        The revisit interval for a confirmation dwell, by default 0.05
    min_revisit_interval : float, optional
        The minimum revisit interval for track updates, by default 0.2
    max_revisit_interval : float, optional
        The maximum revisit interval for track updates, by default 2.0
    position_mapping : List[int], optional
        The indices of the state vector that correspond to position, by default [0, 2, 4]
    velocity_mapping : List[int], optional
        The indices of the state vector that correspond to velocity, by default [1, 3, 5]
    n_confirm_detections : int, optional
        Number of detections that must be associated to a tentative track before it is confirmed, by default 3

    References
    ----------
    [1] "Novel Radar Techniques and Applications", Richard Klemm, 2019
    """
    self.tracker = tracker
    self.track_sharpness = track_sharpness
    self.confirmation_interval = confirmation_interval
    self.min_revisit_interval = min_revisit_interval
    self.max_revisit_interval = max_revisit_interval
    self.position_mapping = position_mapping
    self.velocity_mapping = velocity_mapping
    self.n_confirm_detections = n_confirm_detections

    self.confirmed_tracks = []
    self.tentative_tracks = []
    self.update_times = {}

  def process_detections(self,
                         detections: List[Detection],
                         time: Union[float, datetime.datetime],
                         look: Look = None,
                         ) -> None:
    for detection in detections:
      # Check if the target is already in the list of tracks.
      target_id = detection.groundtruth_path.id
      confirmed_target_ids = [
          track.target_id for track in self.confirmed_tracks]
      tentative_target_ids = [
          track.target_id for track in self.tentative_tracks]

      if target_id in confirmed_target_ids:  # Confirmed track update
        track = self.confirmed_tracks[confirmed_target_ids.index(target_id)]
        state = State(state_vector=track.state_vector,
                      covar=track.covar,
                      timestamp=time,)
        # Incorporate the new measurement into the track history
        state.state_vector, state.covar = self.tracker.predict(state, time)
        state.state_vector, state.covar = self.tracker.update(
            state, detection.state_vector)
        track.append(state)
        dt = adaptive_revisit_interval(
            state_vector=track.state_vector,
            covar=track.covar,
            predict_func=self.tracker.predict_func,
            transition_model=self.tracker.transition_model,
            az_beamwidth=look.azimuth_beamwidth,
            el_beamwidth=look.elevation_beamwidth,
            track_sharpness=self.track_sharpness,
            min_revisit_interval=self.min_revisit_interval,
            max_revisit_interval=self.max_revisit_interval,
        )

      elif target_id in tentative_target_ids:  # Tentative track update
        track = self.tentative_tracks[tentative_target_ids.index(target_id)]
        state = State(state_vector=track.state_vector,
                      covar=track.covar,
                      timestamp=time,)
        # Incorporate the new measurement into the track history
        state.state_vector, state.covar = self.tracker.predict(state, time)
        state.state_vector, state.covar = self.tracker.update(
            state, detection.state_vector)
        track.append(state)
        # Remove tracks from the queue once they've been confirmed
        if len(track) >= self.n_confirm_detections:
          self.confirmed_tracks.append(track)
          self.tentative_tracks.remove(track)
        dt = self.confirmation_interval

      else:  # New track
        track = self.tracker.initiate(detection)
        self.tentative_tracks.append(track)
        dt = self.confirmation_interval

      if isinstance(time, datetime.datetime):
        dt = datetime.timedelta(seconds=dt)
      self.update_times[track.id] = time + dt

  def generate_looks(self, time: Union[float, datetime.datetime]) -> Look:
    confirmed_track_ids = [track.id for track in self.confirmed_tracks]
    tentative_track_ids = [track.id for track in self.tentative_tracks]
    min_update_interval = min(self.confirmation_interval,
                              self.min_revisit_interval)

    new_looks = []
    for track_id, update_time in self.update_times.items():
      # If this was not present, a look would be generated for every track each time this function is called.
      if update_time >= time + min_update_interval:
        continue

      # If we're already late on scheduling the look for this track, schedule it immediately
      start_time = time if time > update_time else update_time

      # Extrack the last state from the track
      # TODO: Set the look priority based on whether or not the track is confirmed.
      if track_id in confirmed_track_ids:
        is_confirmed = True
        itrack = confirmed_track_ids.index(track_id)
        last_state = self.confirmed_tracks[itrack].history[-1]
      else:
        is_confirmed = False
        itrack = tentative_track_ids.index(track_id)
        last_state = self.tentative_tracks[itrack].history[-1]

      # Propagate the state to the current time. Use the predicted position to determine where the beam should be steered.
      predicted_state, _ = self.tracker.predict(last_state, start_time)
      predicted_pos = predicted_state[self.position_mapping].ravel()
      az, el, r = cart2sph(*predicted_pos, degrees=True)

      if is_confirmed:
        look = TrackUpdateLook(azimuth_steering_angle=az,
                               elevation_steering_angle=el,
                               start_time=start_time,
                               )
      else:
        look = TrackConfirmationLook(azimuth_steering_angle=az,
                                     elevation_steering_angle=el,
                                     start_time=start_time,
                                     )
      new_looks.append(look)

    return new_looks


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