import datetime
from typing import Optional

import numpy as np
from mpar_sim.models.measurement.base import MeasurementModel


class Detection():
  def __init__(self,
               measurement: np.array = None,
               measurement_model: MeasurementModel = None,
               snr: float = None,
               timestamp: datetime.datetime = None,
               metadata: dict = None,
               ):
    self.measurement = measurement
    self.measurement_model = measurement_model
    self.snr = snr
    self.timestamp = timestamp
    self.metadata = metadata if metadata else {}
    
class TrueDetection(Detection):
  def __init__(self,
               origin = None,
               **kwargs):
    super().__init__(**kwargs)
    self.origin = origin


class FalseDetection(Detection):
  """A detection due to thermal noise or clutter."""
