import datetime
from typing import Union


class Look():
  pass

# TODO: Add default parameters
class RadarLook(Look):
  def __init__(self,
               # Beam parameters
               azimuth_steering_angle: float = 0,
               elevation_steering_angle: float = 0,
               azimuth_beamwidth: float = 10,
               elevation_beamwidth: float = 10,
               # Waveform parameters
               bandwidth: float = 50e6,
               pulsewidth: float = 10e-6,
               prf: float = 1500,
               n_pulses: int = 10,
               tx_power: float = 1,
               # Scheduler parameters
               start_time: Union[float, datetime.datetime] = 0,
               priority: float = 0,
               
               ) -> None:
    # Beam parameters
    self.azimuth_steering_angle = azimuth_steering_angle
    self.elevation_steering_angle = elevation_steering_angle
    self.azimuth_beamwidth = azimuth_beamwidth
    self.elevation_beamwidth = elevation_beamwidth
    
    # Waveform parameters
    self.bandwidth = bandwidth
    self.pulsewidth = pulsewidth
    self.prf = prf
    self.n_pulses = n_pulses
    self.dwell_time = n_pulses / prf
    
    # Scheduler parameters
    self.start_time = start_time
    self.priority = priority
    
    self.tx_power = tx_power
    
  @property
  def end_time(self) -> Union[float, datetime.datetime]:
    if isinstance(self.start_time, datetime.datetime):
      return self.start_time + datetime.timedelta(seconds=self.dwell_time)
    else:
      return self.start_time + self.dwell_time