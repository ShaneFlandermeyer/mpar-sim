from typing import Set
from stonesoup.types.detection import Detection
from stonesoup.types.track import Track
import datetime


class TWSAgent:
  """A track-while-scan agent."""

  def __init__(self,
               initiator,
               associator,
               updater,
               ):
    self.initiator = initiator
    self.associator = associator
    self.updater = updater
    self.confirmed_tracks = set()

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
        post = self.updater.update(hypothesis)
        track.append(post)
        associated_detections.add(hypothesis.measurement)
      else:
        # When data associator says no detections are good enough, we'll keep the prediction
        track.append(hypothesis.prediction)
    self.confirmed_tracks |= self.initiator.initiate(
        detections=detections - associated_detections,
        timestamp=timestamp)
    return self.confirmed_tracks
