class Look():
  pass


class RadarLook(Look):
  def __init__(self,
               start_time: float,
               # Beam parameters
               azimuth_steering_angle: float,
               elevation_steering_angle: float,
               azimuth_beamwidth: float,
               elevation_beamwidth: float,
               # Waveform parameters
               bandwidth: float,
               pulsewidth: float,
               prf: float,
               n_pulses: int
               ) -> None:
    self.start_time = start_time
    
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