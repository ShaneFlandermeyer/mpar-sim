import datetime
from typing import List, Set

from stonesoup.dataassociator.base import Associator
from stonesoup.deleter.base import Deleter
from stonesoup.initiator.base import Initiator
from stonesoup.types.detection import Detection
from stonesoup.types.track import Track
from stonesoup.updater.base import Updater

from mpar_sim.looks.look import Look


class TWSAgent:
  """A track-while-scan agent."""

  def __init__(self,
               initiator: Initiator,
               associator: Associator,
               updater: Updater,
               deleter: Deleter,
               ):
    # Tracker components
    self.initiator = initiator
    self.associator = associator
    self.updater = updater
    self.deleter = deleter
    
    # Active tracks
    self.confirmed_tracks = set()

  def act(self, current_time: datetime.datetime) -> List[Look]:
    """
    Generate an empty list of looks (since all looks for the track-while-scan agent are generated from the search function)

    Parameters
    ----------
    current_time : datetime.datetime
        Current simulation time

    Returns
    -------
    List[Look]
        Empty list of look requests
    """
    return []

  def update_tracks(self,
                    detections: Set[Detection],
                    timestamp: datetime.datetime) -> Set[Track]:
    """
    Use all detections to update the list of tracks

    Parameters
    ----------
    detections : Set[Detection]
        Detections from search dwell(s)
    timestamp : datetime.datetime
        Current time

    Returns
    -------
    Set[Track]
        Tracks that have been confirmed
    """
    # Calculate all hypothesis pairs and associate the elements in the best subset to the tracks.
    hypotheses = self.associator.associate(self.confirmed_tracks,
                                           detections,
                                           timestamp=timestamp)
    associated_detections = set()
    # Update confirmed tracks with new measurements
    for track in self.confirmed_tracks:
      hypothesis = hypotheses[track]
      if hypothesis.measurement:
        post = self.updater.update(hypothesis, timestamp=timestamp)
        track.append(post)
        associated_detections.add(hypothesis.measurement)
      # elif track in self.confirmed_tracks:
      #   # When data associator says no detections are good enough, we'll keep the prediction
      #   track.append(hypothesis.prediction)
        
    self.confirmed_tracks -= self.deleter.delete_tracks(self.confirmed_tracks)
    self.confirmed_tracks |= self.initiator.initiate(
        detections=detections - associated_detections,
        timestamp=timestamp)
    return self.confirmed_tracks
