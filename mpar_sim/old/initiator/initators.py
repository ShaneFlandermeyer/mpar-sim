import datetime
from typing import List

import numpy as np
from stonesoup.base import Property
from stonesoup.dataassociator.base import DataAssociator
from stonesoup.deleter.base import Deleter
from stonesoup.initiator.base import Initiator
from stonesoup.initiator.simple import (GaussianInitiator,
                                        SimpleMeasurementInitiator)
from stonesoup.models.base import LinearModel, ReversibleModel
from stonesoup.models.measurement import MeasurementModel
from stonesoup.types.detection import Detection
from stonesoup.types.hypothesis import SingleHypothesis
from stonesoup.types.state import GaussianState, State
from stonesoup.types.track import Track
from stonesoup.types.update import GaussianStateUpdate, Update
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
  associator: DataAssociator = Property(
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
      associations = self.associator.associate(
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

class SimpleMeasurementInitiator(GaussianInitiator):
    """Initiator that maps measurement space to state space

    Works for both linear and non-linear co-ordinate input

    This initiator utilises the :class:`~.MeasurementModel` matrix to convert
    :class:`~.Detection` state vector and model covariance into state space.
    It either takes the :class:`~.MeasurementModel` from the given detection
    or uses the :attr:`measurement_model`.

    Utilises the ReversibleModel inverse function to convert
    non-linear spherical co-ordinates into Cartesian x/y co-ordinates
    for use in predictions and mapping.

    This then replaces mapped values in the :attr:`prior_state` to form the
    initial :class:`~.GaussianState` of the :class:`~.Track`.

    The diagonal loading value is used to try to ensure that the estimated
    covariance matrix is positive definite, especially for subsequent Cholesky
    decompositions.
    """
    prior_state: GaussianState = Property(doc="Prior state information")
    measurement_model: MeasurementModel = Property(
        default=None,
        doc="Measurement model. Can be left as None if all detections have a "
            "valid measurement model.")
    skip_non_reversible: bool = Property(default=False)
    diag_load: float = Property(default=0.0, doc="Positive float value for diagonal loading")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.diag_load < 0:
            raise ValueError(
                "diag_load value can't be less than 0.0")

    def initiate(self, detections, timestamp, **kwargs):
        tracks = set()

        for detection in detections:
            if detection.measurement_model is not None:
                measurement_model = detection.measurement_model
            else:
                if self.measurement_model is None:
                    raise ValueError("No measurement model specified")
                else:
                    measurement_model = self.measurement_model

            if isinstance(measurement_model, LinearModel):
                model_matrix = measurement_model.matrix()
                inv_model_matrix = np.linalg.pinv(model_matrix)
                state_vector = inv_model_matrix @ detection.state_vector
            else:
                if isinstance(measurement_model, ReversibleModel):
                    try:
                        state_vector = measurement_model.inverse_function(
                            detection)
                    except NotImplementedError:
                        if not self.skip_non_reversible:
                            raise
                        else:
                            continue
                    model_matrix = measurement_model.jacobian(State(
                        state_vector))
                    inv_model_matrix = np.linalg.pinv(model_matrix)
                elif self.skip_non_reversible:
                    continue
                else:
                    raise Exception("Invalid measurement model used.\
                                    Must be instance of linear or reversible.")

            model_covar = measurement_model.covar()

            prior_state_vector = self.prior_state.state_vector.copy()
            prior_covar = self.prior_state.covar.copy()

            mapped_dimensions = measurement_model.mapping

            prior_state_vector[mapped_dimensions, :] = 0
            prior_covar[mapped_dimensions, :] = 0
            C0 = inv_model_matrix @ model_covar @ inv_model_matrix.T
            C0 = C0 + prior_covar + np.diag(np.array([self.diag_load] * C0.shape[0]))
            tracks.add(Track([GaussianStateUpdate(
                prior_state_vector + state_vector,
                C0,
                SingleHypothesis(None, detection),
                timestamp=detection.timestamp)
            ]))
        return tracks


class MultiMeasurementInitiator(GaussianInitiator):
    """Multi-measurement initiator.

    Utilises features of the tracker to initiate and hold tracks
    temporarily within the initiator itself, releasing them to the
    tracker once there are multiple detections associated with them
    enough to determine that they are 'sure' tracks.

    Utilises simple initiator to initiate tracks to hold ->
    prevents code duplication.

    Solves issue of short-lived single detection tracks being
    initiated only to then be removed shortly after.
    Does cause slight delay in initiation to tracker.
    
    This class is identical to the stonesoup version, but also returns the associated detections for use in the RL environment.
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
    min_points: int = Property(
        default=2, doc="Minimum number of track points required to confirm a track.")
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
            self.initiator = SimpleMeasurementInitiator(self.prior_state, self.measurement_model)

    def initiate(self, detections, timestamp, **kwargs):
        sure_tracks = set()

        associated_detections = set()

        if self.holding_tracks:
            associations = self.data_associator.associate(
                self.holding_tracks, detections, timestamp)

            for track, hypothesis in associations.items():
                if hypothesis:
                    state_post = self.updater.update(hypothesis)
                    track.append(state_post)
                    associated_detections.add(hypothesis.measurement)
                else:
                    track.append(hypothesis.prediction)

                if sum(1 for state in track if not self.updates_only or isinstance(state, Update))\
                        >= self.min_points:
                    sure_tracks.add(track)
                    self.holding_tracks.remove(track)

            self.holding_tracks -= self.deleter.delete_tracks(self.holding_tracks)

        self.holding_tracks |= self.initiator.initiate(
            detections - associated_detections, timestamp)
        return sure_tracks, associated_detections
    
    def reset(self):
        self.holding_tracks = set()