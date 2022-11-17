import datetime
from typing import List

import numpy as np
from stonesoup.base import Property
from stonesoup.dataassociator.base import DataAssociator
from stonesoup.deleter.base import Deleter
from stonesoup.initiator.base import Initiator
from stonesoup.initiator.simple import (GaussianInitiator,
                                        SimpleMeasurementInitiator)
from stonesoup.models.measurement import MeasurementModel
from stonesoup.types.detection import Detection
from stonesoup.types.state import GaussianState
from stonesoup.updater.base import Updater


class MofNInitiator(GaussianInitiator):
  """M-of-N initiator

  Utilizes features of the tracker to initiate and hold tracks temporarily in the initiator itself, releasing them to the tracker when M of the last N detections have been associated with them to determine that they are 'sure' tracks.

  Utilises simple initiator to initiate tracks to hold ->
  prevents code duplication.

  Solves issue of short-lived single detection tracks being
  initiated only to then be removed shortly after.
  Does cause slight delay in initiation to tracker.
  """
  prior_state: GaussianState = Property(doc="Prior state information")
  deleter: Deleter = Property(doc="Deleter used to delete the track.")
  data_associator: DataAssociator = Property(
      doc="Association algorithm to pair predictions to detections.")
  updater: Updater = Property(
      doc="Updater used to update the track object to the new state.")
  measurement_model: MeasurementModel = Property(
      default=None,
      doc="Measurement model. Can be left as None if all detections have a "
          "valid measurement model.")
  confirmation_threshold: List = Property(
      default=[2, 3], doc="Threshold for track confirmation, specified as a 2-element list [M, N]. M of the last N detections must be associated with the track to confirm it.")
  updates_only: bool = Property(
      default=True, doc="Whether :attr:`min_points` only counts :class:`~.Update` states.")
  initiator: Initiator = Property(
      default=None,
      doc="Initiator used to create tracks. If None, a :class:`SimpleMeasurementInitiator` will "
          "be created using :attr:`prior_state` and :attr:`measurement_model`. Otherwise, these "
          "attributes are ignored.")

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.holding_tracks = set()
    if self.initiator is None:
      self.initiator = SimpleMeasurementInitiator(
          self.prior_state, self.measurement_model)

  def initiate(self, detections: set[Detection], timestamp: datetime.datetime):
    sure_tracks = set()

    associated_detections = set()

    if self.holding_tracks:
      # Try to associate new detections to tentative tracks
      associations = self.data_associator.associate(
          self.holding_tracks, detections, timestamp)

      for track, hypothesis in associations.items():
        track._history = np.roll(track._history, 1)
        if hypothesis:
          state_post = self.updater.update(hypothesis)
          track.append(state_post)
          associated_detections.add(hypothesis.measurement)
          track._history[0] = 1
        else:
          track.append(hypothesis.prediction)
          track._history[0] = 0
          
        # TODO: Check for M-of-N threshold
        if np.sum(track._history) >= self.confirmation_threshold[0]:
          sure_tracks.add(track)
          self.holding_tracks.remove(track)
        
      self.holding_tracks -= self.deleter.delete_tracks(self.holding_tracks)
      
    # Initialize new tracks to add to the tentative track list. Each of these tracks need an extra history parameter to handle M-of-N confirmation logic.
    new_tracks = self.initiator.initiate(
      detections - associated_detections, timestamp)
    for track in new_tracks:
      track._history = np.zeros(self.confirmation_threshold[1])
    self.holding_tracks |= new_tracks
      
    
    return sure_tracks
