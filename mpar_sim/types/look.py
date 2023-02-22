import datetime
from typing import Optional, Union


class Look():
  """
  Base class for radar look objects. This class is intended to be used with the PhasedArrayRadar class. The parameters of the look can be loaded into the radar object using the load_look() method.
  """

  def __init__(self,
               # Beam parameters
               azimuth_steering_angle: Optional[float] = None,
               elevation_steering_angle: Optional[float] = None,
               azimuth_beamwidth: Optional[float] = None,
               elevation_beamwidth: Optional[float] = None,
               # Waveform parameters
               center_frequency: Optional[float] = None,
               bandwidth: Optional[float] = None,
               pulsewidth: Optional[float] = None,
               prf: Optional[float] = None,
               n_pulses: Optional[int] = None,
               tx_power: Optional[float] = None,
               # Scheduler parameters
               start_time: Optional[Union[float, datetime.datetime]] = None,
               priority: Optional[float] = None,
               ) -> None:
    # Beam parameters
    self.azimuth_steering_angle = azimuth_steering_angle
    self.elevation_steering_angle = elevation_steering_angle
    self.azimuth_beamwidth = azimuth_beamwidth
    self.elevation_beamwidth = elevation_beamwidth

    # Waveform parameters
    self.center_frequency = center_frequency
    self.bandwidth = bandwidth
    self.pulsewidth = pulsewidth
    self.prf = prf

    self.n_pulses = n_pulses
    if n_pulses is not None:
      self.n_pulses = int(n_pulses)
      self.dwell_time = float(n_pulses / prf)

    # Scheduler parameters
    self.start_time = start_time
    self.priority = priority
    self.tx_power = tx_power


class TrackConfirmationLook(Look):
  """A look for initiating a track for new targets."""


class TrackUpdateLook(Look):
  """A look for updating an existing track"""


class VolumeSearchLook(Look):
  """A look for searching for conducting a volume search step."""
  pass
