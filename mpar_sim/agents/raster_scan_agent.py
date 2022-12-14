import numpy as np
from mpar_sim.job import RadarJob

from mpar_sim.agents.agent import Agent


class RasterScanAgent(Agent):
  """
  An agent that places beams at predetermined locations that cover a volume. 

  Parameters
  ----------
  azimuth_scan_limits : np.ndarray
      Azimuth scan range in degrees
  elevation_scan_limits : np.ndarray
      Elevation scan range in degrees
  azimuth_beam_spacing : float
      Azimuth beam spacing in beamwidths
  elevation_beam_spacing : float
      Elevation beam spacing in beamwidths
  azimuth_beamwidth : float
      Azimuth beamwidth in degrees
  elevation_beamwidth : float
      Elevation beamwidth in degrees
  bandwidth : float, optional
      Waveform bandwidth, by default 5e6
  pulsewidth : float, optional
      Waveform pulsewidth, by default 10e-6
  prf : float, optional
      Pulse repetition frequency, by default 1500
  n_pulses : int, optional
      Number of pulses, by default 1
  """

  def __init__(self,
               azimuth_scan_limits: np.ndarray,
               elevation_scan_limits: np.ndarray,
               azimuth_beam_spacing: float,
               elevation_beam_spacing: float,
               azimuth_beamwidth: float,
               elevation_beamwidth: float,
               bandwidth: float = 5e6,
               pulsewidth: float = 10e-6,
               prf: float = 1500,
               n_pulses: int = 1,
               ):
    self.azimuth_scan_limits = azimuth_scan_limits
    self.elevation_scan_limits = elevation_scan_limits
    self.azimuth_beam_spacing = azimuth_beam_spacing
    self.elevation_beam_spacing = elevation_beam_spacing
    self.azimuth_beamwidth = azimuth_beamwidth
    self.elevation_beamwidth = elevation_beamwidth
    self.bandwidth = bandwidth
    self.pulsewidth = pulsewidth
    self.prf = prf
    self.n_pulses = n_pulses

    # Compute the beam search grid
    d_az = azimuth_beam_spacing*azimuth_beamwidth
    d_el = elevation_beam_spacing*elevation_beamwidth
    az_beam_positions = np.arange(
        azimuth_scan_limits[0], azimuth_scan_limits[1], d_az)
    el_beam_positions = np.arange(
        elevation_scan_limits[0], elevation_scan_limits[1], d_el)

    # Create a grid that contains all possible beam positions
    az_grid, el_grid = np.meshgrid(az_beam_positions, el_beam_positions)
    self.beam_positions = np.stack((
        az_grid.flatten(), el_grid.flatten()), axis=0)
    self.n_positions = self.beam_positions.shape[1]
    self.current_position = 0

  def act(self, current_time: float) -> RadarJob:
    """
    Select a new set of task parameters

    Parameters
    ----------
    current_time: float
      Current time in seconds

    Returns
    -------
    RadarJob
        A new job at the next beam position in the raster scan
    """

    # Select a new beam position
    beam_position = self.beam_positions[:, self.current_position]
    self.current_position = (self.current_position +
                             1) % self.n_positions
    # Create a new job
    job = RadarJob(
        start_time=current_time,
        azimuth_steering_angle=beam_position[0],
        elevation_steering_angle=beam_position[1],
        azimuth_beamwidth=self.azimuth_beamwidth,
        elevation_beamwidth=self.elevation_beamwidth,
        bandwidth=self.bandwidth,
        pulsewidth=self.pulsewidth,
        prf=self.prf,
        n_pulses=self.n_pulses
    )

    return job
