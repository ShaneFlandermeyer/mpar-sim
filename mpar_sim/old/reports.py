import numpy as np
from typing import List, Union


class DetectionReport():
  """
  Stores measurements for detections from a list of targets 
  """

  def __init__(self,
               # Measurement dimensions
               rng: Union[np.ndarray, None] = None,
               velocity: Union[np.ndarray, None] = None,
               azimuth: Union[np.ndarray, None] = None,
               elevation: Union[np.ndarray, None] = None,
               # Measurement resolutions
               range_resolution: Union[float, None] = None,
               velocity_resolution: Union[float, None] = None,
               azimuth_resolution: Union[float, None] = None,
               elevation_resolution: Union[float, None] = None,
               # Additional metadata
               time: Union[float, List[float], None] = None,
               detection_probability: Union[np.ndarray, None] = None,
               snr: Union[np.ndarray, None] = None,
               target_ids: Union[List[int], np.ndarray] = np.array([]),
               ) -> None:
    # Measurement dimensions
    self.range = rng
    self.velocity = velocity
    self.azimuth = azimuth
    self.elevation = elevation
    
    # Measurement resolutions in each dimension
    self.range_resolution = range_resolution
    self.velocity_resolution = velocity_resolution
    self.azimuth_resolution = azimuth_resolution
    self.elevation_resolution = elevation_resolution
    
    # Metadata
    self.time = time
    self.target_ids = target_ids
    self.detection_probability = detection_probability
    self.snr = snr
    self.n_targets = len(self.target_ids)
    
  def __str__(self): 
    return f""" 
    Detection Report:
    -----------------
    Time: {self.time} s
    Range resolution: {self.range_resolution} m
    Velocity resolution: {self.velocity_resolution} m/s
    Azimuth resolution: {self.azimuth_resolution} deg
    Elevation resolution: {self.elevation_resolution} deg
    
    Number of targets detected: {self.n_targets}
    Ranges: {self.range} m
    Velocities: {self.velocity} m/s
    Azimuths: {self.azimuth} deg
    Elevations: {self.elevation} deg
    Target IDs: {self.target_ids}
    Detection probabilities: {self.detection_probability}
    SNRs: {self.snr}
    """
    
  def __len__(self) -> int:
    return self.n_targets

  def merge(self) -> None:
    """
    Merge detections from multiple targets that are in the same measurement bins

    TODO: This function currently just takes the largest measurement from the bin, but it should really compute a centroid from all targets in each measurement dimension 
    """
    n_ambiguous_targets = len(self.target_ids)

    # Convert the list of measurements to a numpy array, where each row is a detection and each column is a measurement dimension
    measurements = np.vstack([x for x in [
                             self.range, self.velocity, self.azimuth, self.elevation] if x is not None]).T
    resolutions = np.array([x for x in [self.range_resolution, self.velocity_resolution,
                           self.azimuth_resolution, self.elevation_resolution] if x is not None])

    # Bin the measurements based on the resolution in that dimension, and only keep the largest measurement in each bin
    bins = np.round(measurements / resolutions)
    _, unique_indices = np.unique(bins, axis=0, return_index=True)

    # Update measurements
    self.remove(np.setdiff1d(np.arange(n_ambiguous_targets), unique_indices))

  def remove(self, indices: np.ndarray):
    """
    Delete the detections at the given indices from the object report.

    Args:
        indices (np.ndarray): Array of detection indices.
    """
    self.range = np.delete(self.range, indices)
    self.velocity = np.delete(self.velocity, indices)
    self.azimuth = np.delete(self.azimuth, indices)
    self.elevation = np.delete(self.elevation, indices)
    self.target_ids = np.delete(self.target_ids, indices)
    self.detection_probability = np.delete(self.detection_probability, indices)
    self.snr = np.delete(self.snr, indices)
    self.n_targets = len(self.target_ids)
