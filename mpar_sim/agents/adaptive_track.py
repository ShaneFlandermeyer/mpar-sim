import datetime
from collections import deque
from datetime import timedelta
from typing import List, Set, Tuple

import numpy as np
from stonesoup.dataassociator.base import Associator
from stonesoup.deleter.base import Deleter
from stonesoup.initiator.base import Initiator
from stonesoup.predictor.base import Predictor
from stonesoup.types.detection import Detection
from stonesoup.types.track import Track
from stonesoup.types.update import Update
from stonesoup.updater.base import Updater

from mpar_sim.agents.agent import Agent
from mpar_sim.common.coordinate_transform import cart2sph, cart2sph_covar
from mpar_sim.looks.look import Look


class AdaptiveTrackAgent(Agent):
  """
  This agent selects beam positions to confirm and update tracks. 

  The agent generates a confirmation look when the tracker processes a detection that has not been associated with any existing tracks. In this case, confirmation looks are scheduled at a high revisit rate and an M-of-N threshold is used to determine if the track should be initialized
  """

  def __init__(self,
               # Tracker components
               initiator: Initiator,
               associator: Associator,
               predictor: Predictor,
               updater: Updater,
               deleter: Deleter,
               # Adaptive tracking parameters
               track_sharpness: float,
               min_revisit_rate: float,
               max_revisit_rate: float,
               confirm_rate: float,
               # Task parameters
               azimuth_beamwidth: float,
               elevation_beamwidth: float,
               bandwidth: float,
               pulsewidth: float,
               prf: float,
               n_pulses: float,
               #
               position_mapping: Tuple = (0, 2, 4),
               ):
    # Tracker components
    self.initiator = initiator
    self.associator = associator
    self.predictor = predictor
    self.updater = updater
    self.deleter = deleter

    # Adaptive tracking parameters
    self.confirm_rate = confirm_rate
    self.min_revisit_rate = min_revisit_rate
    self.max_revisit_rate = max_revisit_rate
    self.track_sharpness = track_sharpness
    self.position_mapping = position_mapping

    # Task parameters
    # TODO: These should depend on the target and task (e.g., confirm vs. update)
    self.azimuth_beamwidth = azimuth_beamwidth
    self.elevation_beamwidth = elevation_beamwidth
    self.bandwidth = bandwidth
    self.pulsewidth = pulsewidth
    self.prf = prf
    self.n_pulses = n_pulses

    # Compute intermediate revisit times
    tmin = 1 / max_revisit_rate
    tmax = 1 / min_revisit_rate
    n = int(np.ceil(np.log(tmax/tmin)/np.log(2)))
    self.revisit_times = tmin * np.power(2, np.arange(n-1))
    self.revisit_times = np.append(self.revisit_times, tmax)

    # A queue of (time, track) tuples that indicates the next desired update time for each track that has received a new detection
    self.update_queue = deque()

    self.confirmed_tracks = set()

  def act(self, current_time: datetime.datetime) -> List[Look]:
    """
    Return a list of looks to be scheduled.

    Parameters
    ----------
    current_time : datetime.datetime
        _description_

    Returns
    -------
    List[Look]
        _description_
    """
    looks = []
    n_looks = len(self.update_queue)
    for _ in range(n_looks):
      next_update_time, track = self.update_queue.pop()
      if next_update_time > current_time:
        start_time = next_update_time
      else:
        start_time = current_time

      # Use the predicted position of the target at the start time to steer the beam
      predicted_state = self.predictor.predict(
          track, timestamp=start_time)
      position_xyz = predicted_state.state_vector[self.position_mapping, :]
      predicted_az, predicted_el, _ = cart2sph(*position_xyz)
      # Create the look
      # TODO: Confirm/update should be different types of looks
      look = Look(
          start_time=start_time,
          azimuth_steering_angle=np.rad2deg(predicted_az),
          elevation_steering_angle=np.rad2deg(predicted_el),
          azimuth_beamwidth=self.azimuth_beamwidth,
          elevation_beamwidth=self.elevation_beamwidth,
          bandwidth=self.bandwidth,
          pulsewidth=self.pulsewidth,
          prf=self.prf,
          n_pulses=self.n_pulses,
          priority=1,
      )
      looks.append(look)

    return looks

  def update_tracks(self,
                    detections: Set[Detection],
                    current_time: datetime.datetime) -> Set[Track]:
    """
    Updates the state of any tracks (tentative or confirmed) that have had a detection assigned to them on this time step. 

    For each updated track, the next update is also scheduled (a confirmation dwell for tentative tracks and an adaptive revisit dwell for confirmed tracks).

    Parameters
    ----------
    detections : Set[Detection]
        New detections made on this time step
    current_time : datetime.datetime
        Start time of the dwell that produced the detections

    Returns
    -------
    Set[Track]
        Confirmed tracks after accounting for the new detections.
    """
    # Associate detections with tracks
    all_tracks = self.confirmed_tracks | self.initiator.holding_tracks
    hypotheses = self.associator.associate(
        all_tracks, detections, timestamp=current_time)

    # Update tracks and schedule new dwells
    associated_detections = set()
    for track in all_tracks:
      hypothesis = hypotheses[track]
      if hypothesis.measurement:
        # Update the track
        posterior = self.updater.update(hypothesis)
        track.append(posterior)

        # Determine the revisit interval
        if track in self.confirmed_tracks:
          # Detection already associated with a confirmed track
          associated_detections.add(hypothesis.measurement)
          # Schedule a new dwell at the adaptive revisit interval
          revisit_interval = self.compute_revisit_interval(
              track, current_time, self.predictor)
          dt = timedelta(seconds=revisit_interval)
        else:
          # Schedule a rapid confirmation dwell
          dt = timedelta(seconds=1 / self.confirm_rate)
        next_update_time = current_time + dt
        # Add the track to the update queue
        self.update_queue.append((next_update_time, track))
        
      elif track in self.confirmed_tracks:
        if current_time > track.states[-1].timestamp:
          track.append(hypothesis.prediction)

    # Try to initiate new tracks from detections that were not associated with any existing tracks
    self.confirmed_tracks -= self.deleter.delete_tracks(self.confirmed_tracks)
    self.confirmed_tracks |= self.initiator.initiate(
        detections=detections - associated_detections,
        timestamp=current_time)

    return self.confirmed_tracks

  def compute_revisit_interval(self,
                               track: Update,
                               current_time: datetime.datetime,
                               predictor: Predictor) -> float:
    """
    Compute the minimum revisit interval for a track

    Parameters
    ----------
    track : Track
        Track whose revisit interval is to be computed
    current_time : datetime.datetime
        Current simulation time

    Returns
    -------
    float
        Revisit interval in seconds
    """
    for revisit_time in reversed(self.revisit_times):
      # Propagate the track forward to the revisit time
      predicted_state = predictor.predict(track,
                                          timestamp=current_time + timedelta(seconds=revisit_time))

      # Extract the state vector and covariance for the cartesian position. Since we're doing adaptive tracking in terms of the track sharpness in angle, we need to convert the covariance from cartesian to spherical
      position_xyz = predicted_state.state_vector[self.position_mapping, :]
      position_covar_xyz = predicted_state.covar[self.position_mapping,
                                                 :][:, self.position_mapping]
      position_covar_sph = cart2sph_covar(position_covar_xyz, *position_xyz)

      # Compute the error standard deviation in azimuth, elevation, and range. With this, the revisit interval is the longest time we can wait before the error exceeds the threshold in
      error_std_dev = np.sqrt(np.diag(position_covar_sph))
      az_error, el_error, _ = error_std_dev
      az_threshold = self.track_sharpness * np.deg2rad(self.azimuth_beamwidth)
      el_threshold = self.track_sharpness * \
          np.deg2rad(self.elevation_beamwidth)
      if az_error < az_threshold and el_error < el_threshold:
        return revisit_time

    # If the track error is never within the limits, return the minimum revisit interval
    return revisit_time
