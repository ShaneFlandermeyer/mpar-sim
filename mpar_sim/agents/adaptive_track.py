from typing import List, Tuple
from mpar_sim.agents.agent import Agent
from mpar_sim.look import RadarLook
import datetime
from datetime import timedelta
from mpar_sim.common.coordinate_transform import cart2sph, cart2sph_covar
import numpy as np
from stonesoup.types.track import Track
from stonesoup.types.update import Update
from stonesoup.predictor.base import Predictor


class AdaptiveTrackAgent(Agent):
  """
  This agent selects beam positions to confirm and update tracks. 

  The agent generates a confirmation look when the tracker processes a detection that has not been associated with any existing tracks. In this case, confirmation looks are scheduled at a high revisit rate and an M-of-N threshold is used to determine if the track should be initialized
  """

  def __init__(self,
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
    # TODO: Don't hard-code this
    self.tx_power = 100e3

    # Compute intermediate revisit times
    tmin = 1 / max_revisit_rate
    tmax = 1 / min_revisit_rate
    n = int(np.ceil(np.log(tmax/tmin)/np.log(2)))
    self.revisit_times = tmin * np.power(2, np.arange(n-1))
    self.revisit_times = np.append(self.revisit_times, tmax)

  def act(self, current_time: datetime.datetime) -> RadarLook:
    """
    Select a look to observe

    Parameters
    ----------
    current_time : datetime.datetime
        _description_

    Returns
    -------
    RadarLook
        _description_
    """
    pass

  def request_looks(self,
                    confirmed_tracks,
                    tentative_tracks,
                    current_time,
                    predictor) -> List[RadarLook]:
    """
    Loop through all tracks that have been assigned a measurement on this time step. If the track has been confirmed, schedule its next update according to the adaptive revisit interval algorithm defined below. Otherwise, schedule a confirmation look in the very near future.

    Parameters
    ----------
    detections : _type_
        _description_
    time : _type_
        _description_
    """
    looks = []
    for track in confirmed_tracks | tentative_tracks:
      # Choose the desired time of the look to be scheduled for this track
      # TODO: Confirmation and revisit dwells may require different beam layouts or look types
      if track in confirmed_tracks:
        revisit_interval = self.compute_revisit_interval(track, current_time)
        next_update_time = current_time + timedelta(revisit_interval)
      else:
        next_update_time = current_time + timedelta(1/self.confirm_rate)

      # Use the predicted az/el of the target at the next update time as the beam center
      predicted_state = predictor.predict(track, timestamp=next_update_time)
      position_xyz = predicted_state.state_vector[self.position_mapping, :]
      # TODO: This may not be in the correct coordinate frame
      predicted_az, predicted_el, _ = cart2sph(*position_xyz)

      # Create the look
      look = RadarLook(
          start_time=next_update_time,
          azimuth_steering_angle=np.rad2deg(predicted_az),
          elevation_steering_angle=np.rad2deg(predicted_el),
          azimuth_beamwidth=self.azimuth_beamwidth,
          elevation_beamwidth=self.elevation_beamwidth,
          bandwidth=self.bandwidth,
          pulsewidth=self.pulsewidth,
          prf=self.prf,
          n_pulses=self.n_pulses,
      )
      looks.append(look)
      
    return looks

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


if __name__ == '__main__':
  agent = AdaptiveTrackAgent(
      track_sharpness=0.05,
      min_revisit_rate=0.5,
      max_revisit_rate=5,
      confirm_rate=20)
  print(agent.revisit_times)
