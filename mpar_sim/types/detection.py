import datetime
from typing import Optional

import numpy as np
from mpar_sim.models.measurement.base import MeasurementModel
from mpar_sim.types.groundtruth import GroundTruthPath


class Detection():
  def __init__(self,
               state_vector: np.ndarray = None,
               snr: Optional[float] = None,
               timestamp: Optional[datetime.datetime] = None,
               measurement_model: Optional[MeasurementModel] = None,
               metadata: Optional[dict] = {},
               **kwargs
               ):
    self.state_vector = state_vector
    self.snr = snr
    self.timestamp = timestamp
    self.measurement_model = measurement_model
    self.metadata = metadata


class TrueDetection(Detection):
  def __init__(self,
               groundtruth_path: Optional[GroundTruthPath] = None,
               **kwargs,
               ):
    super().__init__(**kwargs)
    self.groundtruth_path = groundtruth_path


class Clutter(Detection):
  """A detection due to thermal noise or clutter."""
