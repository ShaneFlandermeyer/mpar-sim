import datetime

import numpy as np
from mpar_sim.models.measurement.base import MeasurementModel
from mpar_sim.types.groundtruth import GroundTruthPath


class Detection():
  def __init__(self,
               state_vector: np.ndarray = None,
               snr: float = None,
               timestamp: datetime.datetime = None,
               measurement_model: MeasurementModel = None,
               metadata: dict = {},
               **kwargs
               ):
    self.state_vector = state_vector
    self.snr = snr
    self.timestamp = timestamp
    self.measurement_model = measurement_model
    self.metadata = metadata


class TrueDetection(Detection):
  def __init__(self,
               groundtruth_path: GroundTruthPath = None,
               **kwargs,
               ):
    super().__init__(**kwargs)
    self.groundtruth_path = groundtruth_path


class Clutter(Detection):
  """A detection due to thermal noise or clutter."""
