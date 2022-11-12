import numpy as np
from typing import Callable, Union, Tuple
from mpar_sim.common.wrap_to_interval import wrap_to_interval


###############################################
# Helper functions
###############################################

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


def beam_broadening_factor(az_steering_angle: float, el_steering_angle: float) -> Tuple[float, float]:
  """
  Computes the beam broadening factor in azimuth and elevation
  Returns:
      Tuple[float, float]: _description_
  """
  # Wrap the scan angle between -180 and 180 degrees
  look_angles = np.array(wrap_to_interval(
      np.array([az_steering_angle, el_steering_angle]), -180, 180), copy=False)

  # The beam broadening factor is approximately equal to the reciprocal of the effective aperture length in each dimension. For a uniform rectangular array, the effective length is proportional to 1/(cos(az)*cos(el)) in azimuth and 1/cos(el) in elevation.
  broadening_az = 1 / \
      (np.cos(np.radians(look_angles[0]))
       * np.cos(np.radians(look_angles[1])))
  broadening_el = 1 / np.cos(np.radians(look_angles[1]))
  return max(broadening_az, np.cos(np.radians(89))), max(broadening_el, np.cos(np.radians(89)))


def beamwidth2gain(azimuth_beamwidth: float,
                   elevation_beamwidth: float,
                   directivity_beamwidth_prod: float = 32400
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
  wavelength: float
      Transmit wavelength (in meters)
  Returns
  -------
  float
      Antenna aperture (in m^2)
  """
  aperture_xy = wavelength / np.deg2rad(beamwidth)
  return aperture_xy

###############################################
# Objects
###############################################


class Beam():

  """
  A radar beam.

  Parameters
  ----------
  azimuth_beamwidth : float
      Azimuth beamwidth (before broadening)
  elevation_beamwidth : float
      Elevation beamwidth (before broadening)
  azimuth_steering_angle : float, optional
      Azimuth steering angle in degrees, by default 0
  elevation_steering_angle : float, optional
      Elevation steering angle in degrees, by default 0
  has_scan_loss : bool, optional
      If true, effects of steering off boresight are included in beamwidth and gain/directivity computations, by default False
  """

  directivity_beamwidth_prod = 26000

  def __init__(self,
               azimuth_beamwidth: float,
               elevation_beamwidth: float,
               azimuth_steering_angle: float = 0,
               elevation_steering_angle: float = 0) -> None:

    self.azimuth_steering_angle = azimuth_steering_angle
    self.elevation_steering_angle = elevation_steering_angle
    self.azimuth_beamwidth = azimuth_beamwidth
    self.elevation_beamwidth = elevation_beamwidth

    # Compute gain for the given beamwidth
    gain_db = beamwidth2gain(
        self.azimuth_beamwidth, self.elevation_beamwidth, self.directivity_beamwidth_prod)
    self.gain = 10**(gain_db/10)


class RectangularBeam(Beam):
  """
  Define a beam with a rectangular power pattern.

  The rectangular pattern has magnitude 1 inside the antenna's field of view and 0 elsewhere

    Parameters
    ----------
    azimuth_beamwidth : float
        Azimuth beamwidth (before broadening)
    elevation_beamwidth : float
        Elevation beamwidth (before broadening)
    azimuth_steering_angle : float, optional
        Azimuth steering angle in degrees, by default 0
    elevation_steering_angle : float, optional
        Elevation steering angle in degrees, by default 0
    has_scan_loss : bool, optional
        If true, effects of steering off boresight are included in beamwidth and gain/directivity computations, by default False
  """

  # Beamwidth-directivity product used for array gain calculations
  directivity_beamwidth_prod = 41253

  def __init__(self, *args, **kwargs) -> None:
    super().__init__(*args, **kwargs)

  def shape_loss(self,
                 az: Union[float, np.ndarray],
                 el: Union[float, np.ndarray]
                 ) -> Union[float, np.ndarray]:
    """
    Compute the loss due to the shape of the beam. For a rectangular element, there is no loss within the beam and infinite loss outside it.

    Args:
        az (Union[float, np.ndarray]): Target azimuth angles
        el (Union[float, np.ndarray]): Target elevation angles
    Returns:
        Union[float, np.ndarray]: Beam shape loss (dB)
    """
    loss = np.zeros_like(az)
    loss[np.logical_or(np.abs(az) > self.azimuth_beamwidth/2, np.abs(el) > self.elevation_beamwidth/2)] = np.inf
    return loss


class GaussianBeam(Beam):
  """
  Define a beam with a Gaussian power pattern.

  See https://www.mathworks.com/help/phased/ref/phased.gaussianantennaelement-system-object.html

    Parameters
    ----------
    azimuth_beamwidth : float
        Azimuth beamwidth (before broadening)
    elevation_beamwidth : float
        Elevation beamwidth (before broadening)
    azimuth_steering_angle : float, optional
        Azimuth steering angle in degrees, by default 0
    elevation_steering_angle : float, optional
        Elevation steering angle in degrees, by default 0
    has_scan_loss : bool, optional
        If true, effects of steering off boresight are included in beamwidth and gain/directivity computations, by default False
  """
  directivity_beamwidth_prod = 32400

  def __init__(self, *args, **kwargs) -> None:
    super().__init__(*args, **kwargs)

  def shape_loss(self,
                 az: Union[float, np.ndarray],
                 el: Union[float, np.ndarray]
                 ) -> Union[float, np.ndarray]:
    """
    Compute the loss due to the shape of the beam.

    See https://www.mathworks.com/help/radar/ref/beamloss.html

    Args:
        az (Union[float, np.ndarray]): Target azimuth angles
        el (Union[float, np.ndarray]): Target elevation angles
    Returns:
        Union[float, np.ndarray]: Beam shape loss (dB)
    """
    beam_shape_loss_az = np.exp(-4*np.log(2) *
                                (az / self.azimuth_beamwidth)**2)
    beam_shape_loss_el = np.exp(-4*np.log(2) *
                                (el / self.elevation_beamwidth)**2)
    beam_shape_loss_db = -10*np.log10(beam_shape_loss_az*beam_shape_loss_el)
    return beam_shape_loss_db
