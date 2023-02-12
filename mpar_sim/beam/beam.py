import numpy as np
from typing import Callable, Union, Tuple
from mpar_sim.beam.common import beamwidth2aperture, beamwidth2gain
from mpar_sim.common.wrap_to_interval import wrap_to_interval


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
               wavelength: float,
               azimuth_beamwidth: float,
               elevation_beamwidth: float,
               azimuth_steering_angle: float = 0,
               elevation_steering_angle: float = 0) -> None:

    self.wavelength = wavelength
    self.azimuth_steering_angle = azimuth_steering_angle
    self.elevation_steering_angle = elevation_steering_angle
    self.azimuth_beamwidth = azimuth_beamwidth
    self.elevation_beamwidth = elevation_beamwidth
    gain_db = beamwidth2gain(
        self.azimuth_beamwidth, self.elevation_beamwidth, self.directivity_beamwidth_prod)
    self.gain = 10**(gain_db/10)

  def norm_pattern_gain(self,
                        az: Union[float, np.ndarray],
                        el: Union[float, np.ndarray]
                        ) -> Union[float, np.ndarray]:
    raise NotImplementedError

  def shape_loss(self,
                 az: Union[float, np.ndarray],
                 el: Union[float, np.ndarray]
                 ) -> Union[float, np.ndarray]:
    raise NotImplementedError


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

  def norm_pattern_gain(self,
                        az: Union[float, np.ndarray],
                        el: Union[float, np.ndarray]
                        ) -> Union[float, np.ndarray]:
    gain = np.zeros_like(az)
    gain[np.logical_or(np.abs(az) <= self.azimuth_beamwidth/2,
                       np.abs(el) <= self.elevation_beamwidth/2)] = 1
    return gain

  def shape_loss(self,
                 az: Union[float, np.ndarray],
                 el: Union[float, np.ndarray]
                 ) -> Union[float, np.ndarray]:
    """
    Compute the loss due to the shape of the beam. For a rectangular element, there is no loss within the beam and infinite loss outside it.

    Parameters
    ----------
    az : Union[float, np.ndarray]
        Azimuth angles
    el : Union[float, np.ndarray]
        Elevation angles
    Returns
    -------
    Union[float, np.ndarray]
        Beam shape loss (dB)
    """
    loss = np.zeros_like(az)
    loss[np.logical_or(np.abs(az) > self.azimuth_beamwidth/2,
                       np.abs(el) > self.elevation_beamwidth/2)] = np.inf
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

    Parameters
    ----------
    az : Union[float, np.ndarray]
        Azimuth angles
    el : Union[float, np.ndarray]
        Elevation angles
    Returns
    -------
    Union[float, np.ndarray]
        Beam shape loss (dB)
    """
    az_pattern_gain = np.exp(-4*np.log(2) *
                             (az / self.azimuth_beamwidth)**2)
    el_pattern_gain = np.exp(-4*np.log(2) *
                             (el / self.elevation_beamwidth)**2)
    gain = az_pattern_gain*el_pattern_gain
    return -10*np.log10(gain)


class SincBeam(Beam):

  directivity_beamwidth_prod = 26000

  def __init__(self,
               *args,
               **kwargs) -> None:
    super().__init__(*args, **kwargs)

  def norm_pattern_gain(self,
                        az: Union[float, np.ndarray],
                        el: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    # All the angles must be raveled to avoid broadcasting issues
    if isinstance(az, np.ndarray):
      az = az.ravel()
    else:
      az = np.array([az])
    if isinstance(el, np.ndarray):
      el = el.ravel()
    else:
      el = np.array([el])

    angles = np.atleast_2d(np.array([az, el])).T
    steering_angles = np.array([self.azimuth_steering_angle,
                                self.elevation_steering_angle])[np.newaxis, :]
    beamwidths = np.array([self.azimuth_beamwidth, self.elevation_beamwidth])[
        np.newaxis, :]
    d_norm = beamwidth2aperture(beamwidths, self.wavelength) / self.wavelength
    gains = np.sinc(d_norm * (np.sin(np.deg2rad(angles)) -
                    np.sin(np.deg2rad(steering_angles))))
    norm_pattern_gain = np.abs(gains[:, 0] * gains[:, 1])
    return norm_pattern_gain if norm_pattern_gain.size > 1 else norm_pattern_gain.item()

  def shape_loss(self,
                 az: Union[float, np.ndarray],
                 el: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Compute the off-boresight pattern loss for a sinc beam pattern.

    Parameters
    ----------
    az : Union[float, np.ndarray]
        Azimuth angle(s)
    el : Union[float, np.ndarray]
        Elevation angle(s)
    Returns
    -------
    Union[float, np.ndarray]
        Beam shape loss (dB)
    """
    power_pattern = self.norm_pattern_gain(az, el)**2
    shape_loss = -10*np.log10(power_pattern)
    return shape_loss


if __name__ == '__main__':
  import scipy.constants as sc

  az_bw = el_bw = 5
  fc = 3e9
  lambda_ = sc.c/fc
  beam = SincBeam(lambda_, az_bw, el_bw)
  beam.shape_loss(0, 0)
  beam.shape_loss(np.zeros((2,)), np.ones((2,)))
  beam.shape_loss(np.zeros((5, 1)), np.ones((5, 1)))
