import datetime
from typing import Union


class Look():
  def __init__(self,
               # Beam parameters
               azimuth_steering_angle: float = 0,
               elevation_steering_angle: float = 0,
               azimuth_beamwidth: float = 10,
               elevation_beamwidth: float = 10,
               # Waveform parameters
               center_frequency: float = 3e9,
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
    self.center_frequency = center_frequency
    self.bandwidth = bandwidth
    self.pulsewidth = pulsewidth
    self.prf = prf
    self.n_pulses = int(n_pulses)
    self.dwell_time = float(n_pulses / prf)
    
    # Scheduler parameters
    self.start_time = start_time
    self.priority = priority
    
    self.tx_power = tx_power