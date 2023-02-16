from typing import Tuple, Union
import numpy as np

from mpar_sim.common.wrap_to_interval import wrap_to_interval


def beam_broadening_factor(az_steering_angle: float, el_steering_angle: float) -> Tuple[float, float]:
  """
  Computes the beam broadening factor in azimuth and elevation
  Returns:
      Tuple[float, float]: _description_
  """
  # Wrap the scan angle between -180 and 180 degrees
  look_angles = np.array(wrap_to_interval(
      np.array([az_steering_angle, el_steering_angle]), -180, 180))
  look_angles = np.deg2rad(look_angles)

  # The beam broadening factor is approximately equal to the reciprocal of the effective aperture length in each dimension. For a uniform rectangular array, the effective length is proportional to 1/(cos(az)*cos(el)) in azimuth and 1/cos(el) in elevation.
  broadening_az = abs(1 / (np.cos(look_angles[0]) * np.cos(look_angles[1])))
  broadening_el = abs(1 / np.cos(look_angles[1]))

  # According to this model, the loss at 90 degrees is infinite. To avoid this, cap the loss at 89 degrees
  max_broadening = 1 / np.cos(np.deg2rad(89))

  return min(broadening_az, max_broadening), min(broadening_el, max_broadening)


def beamwidth2gain(azimuth_beamwidth: float,
                   elevation_beamwidth: float,
                   directivity_beamwidth_prod: float = 26e3
                   ) -> float:
  """
  Computes the antenna gain (in dBi) from azimuth and elevation beamwidths (in degrees)

  See https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=730532

  Parameters
  ----------
  azimuth_beamwidth : float
      Azimuth beamwidth (in degrees)
  elevation_beamwidth : float
      Elevation beamwidth (in degrees)
  directivity_beamwidth_prod : float
      Numerator for the gain calculation. Defaults to 26000

  Returns
  -------
  float
      Antenna gain (in dBi)
  """
  G = directivity_beamwidth_prod / (azimuth_beamwidth * elevation_beamwidth)
  return 10 * np.log10(G)


def aperture2beamwidth(
    d: Union[np.ndarray, float],
    wavelength: float
) -> Union[np.ndarray, float]:
  """
  Compute the half-power beamwidth from the aperture length
  Parameters
  ----------
  d: Union[np.ndarray, float]
      Aperture lengths in m
  wavelength : float
      Radar wavelength in m
  Returns
  -------
  float
      Beamwidths for each aperture/wavelength pair
  """
  return np.rad2deg(0.886 * wavelength / d)


def beamwidth2aperture(
        beamwidth: Union[np.ndarray, float],
        wavelength: float
) -> float:
  """
  Computes the antenna aperture in each dimension from the beamwidths (in meters) and wavelength (in meters)
  Parameters
  ----------
  beamwidth : Union[np.ndarray, float]
      Beamwidth in each dimension (in degrees)
  wavelength: float
      Transmit wavelength (in meters)
  Returns
  -------
  float
      Antenna aperture (in m^2)
  """
  d = 0.886*wavelength / np.deg2rad(beamwidth)
  return d


def beam_scan_loss(az_steering_angle: float, el_steering_angle: float, az_cosine_power: float = 2, el_cosine_power: float = 2) -> float:
  """
  Compute the loss due to the scan angle. The loss is proportional to the cosine of the scan angle in each dimension.

  Args:
      az_steering_angle (float): Azimuth scan angle
      el_steering_angle (float): Elevation scan angle
      az_cosine_power (float, optional): Power of the cosine in azimuth. Defaults to 2.
      el_cosine_power (float, optional): Power of the cosine in elevation. Defaults to 2.

  Returns:
      float: Scan loss (dB)
  """
  # Wrap the scan angle between -180 and 180 degrees
  return 10 * np.log10(np.power(np.cos(np.radians(az_steering_angle)), az_cosine_power) *
                       np.power(np.cos(np.radians(el_steering_angle)), el_cosine_power))
