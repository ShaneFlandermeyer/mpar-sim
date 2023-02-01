from functools import lru_cache
import numpy as np
from typing import Tuple, Union


def azel2uv(az: np.ndarray, el: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
  """
  Converts az/el in DEGREES to UV 

  Args:
      az (np.ndarray): Azimuth values
      el (np.ndarray): Elevation values

  Returns:
      Tuple(np.ndarray): The tuple (u,v)
  """

  u = np.cos(np.radians(el)) * np.sin(np.radians(az))
  v = np.sin(np.radians(el))

  return (u, v)


def sph2cart(
    azimuth: Union[float, np.ndarray],
    elevation: Union[float, np.ndarray],
    range: Union[float, np.ndarray],
    degrees: bool = False
) -> Union[Tuple[float, float, float], Tuple[np.ndarray, np.ndarray, np.ndarray]]:
  """
  Convert spherical coordinates to cartesian

  Parameters
  ----------
  azimuth: Union[float, np.ndarray]: 
    Azimuth angle in radians
  elevation: Union[float, np.ndarray]: 
    Elevation angle
  range: Union[float, np.ndarray]: 
    Range
  degrees: bool

  Returns
  -------
      Union[float, np.ndarray]: Cartesian coordinates
  """
  if degrees:
    azimuth = np.deg2rad(azimuth)
    elevation = np.deg2rad(elevation)
  x = range * np.cos(elevation) * np.cos(azimuth)
  y = range * np.cos(elevation) * np.sin(azimuth)
  z = range * np.sin(elevation)

  return (x, y, z)


def cart2sph(
    x: Union[float, np.ndarray],
    y: Union[float, np.ndarray],
    z: Union[float, np.ndarray],
    degrees: bool = False
) -> Union[Tuple[float, float, float], Tuple[np.ndarray, np.ndarray, np.ndarray]]:
  """
  Convert cartesian coordinates to spherical

  Parameters
  ----------
  x: Union[float, np.ndarray]
    X coordinate
  y: Union[float, np.ndarray]
    Y coordinate
  z: Union[float, np.ndarray]
    Z coordinate
  degrees: bool
    Return az/el in degrees if true and radians if false, default is false

  Returns
  -------
      Union[float, np.ndarray]: Spherical coordinates
  """
  azimuth = np.arctan2(y, x)
  elevation = np.arctan2(z, np.sqrt(x**2 + y**2))
  r = np.sqrt(x**2 + y**2 + z**2)

  if degrees:
    azimuth = np.rad2deg(azimuth)
    elevation = np.rad2deg(elevation)

  return (azimuth, elevation, r)


def azel2rotmat(az: float, el: float) -> np.ndarray:
  """
  Convert azimuth and elevation to rotation matrix.

  See: http://eecs.qmul.ac.uk/~gslabaugh/publications/euler.pdf

  Args:
      az (float): Azimuth angle (degrees)
      el (float): Elevation angle (degrees)

  Returns:
      np.ndarray: Rotation matrix
  """

  # convert to radians
  az = np.radians(az)
  # Note: The elevation angle used for radar has opposite sign compared to the angle used to define rotation about the y-axis in the source above
  el = -np.radians(el)
  # rotation matrix
  R = np.array([[np.cos(az)*np.cos(el), -np.sin(az), np.cos(az)*np.sin(el)],
                [np.sin(az)*np.cos(el), np.cos(az), np.sin(az)*np.sin(el)],
                [-np.sin(el), 0, np.cos(el)]])
  return R


def rotx(theta: Union[float, np.ndarray]):
  """
  Rotation matrix about the x-axis

  Parameters
  ----------
  theta : Union[float, np.ndarray]
      rotation angle (radians)
  """
  cos_theta, sin_theta = np.cos(theta), np.sin(theta)
  zeros = np.zeros_like(theta)
  ones = np.ones_like(theta)
  return np.array([[ones, zeros, zeros],
                   [zeros, cos_theta, -sin_theta],
                   [zeros, sin_theta, cos_theta]])


def roty(theta: Union[float, np.ndarray]):
  """
  Rotation matrix about the y-axis

  Parameters
  ----------
  theta : Union[float, np.ndarray]
      rotation angle (radians)
  """
  cos_theta, sin_theta = np.cos(theta), np.sin(theta)
  zeros = np.zeros_like(theta)
  ones = np.ones_like(theta)
  return np.array([[cos_theta, zeros, sin_theta],
                   [zeros, ones, zeros],
                   [-sin_theta, zeros, cos_theta]])


def rotz(theta: Union[float, np.ndarray]):
  """
  Rotation matrix about the z-axis

  Parameters
  ----------
  theta : Union[float, np.ndarray]
      rotation angle (radians)
  """
  cos_theta, sin_theta = np.cos(theta), np.sin(theta)
  zeros = np.zeros_like(theta)
  ones = np.ones_like(theta)
  return np.array([[cos_theta, -sin_theta, zeros],
                   [sin_theta, cos_theta, zeros],
                   [zeros, zeros, ones]])

@lru_cache()
def rpy2rotmat(roll: float, pitch: float, yaw: float) -> np.ndarray:
  """
  Convert roll, pitch, yaw to rotation matrix

  Parameters
  ----------
  r : float
      roll angle (radians)
  p : float
      pitch angle (radians)
  y : float
      yaw angle (radians)

  Returns
  -------
  np.ndarray
      Rotation matrix
  """
  R_roll = rotx(roll)
  R_pitch = roty(pitch)
  R_yaw = rotz(yaw)
  return R_yaw @ R_pitch @ R_roll


def cart2sph_covar(cart_covar: np.ndarray, x: float, y: float, z: float) -> np.ndarray:
  """
    Convert a covariance matrix in cartesian coordinates to spherical coordinates

    Parameters
    ----------
    cart_cov : np.ndarray
        Cartesian covariance matrix
    x : float
        Position x-coordinate
    y : float
        Position y-coordinate
    z : float
        Position z-coordinate

    Returns
    -------
    np.ndarray
        Covariance matrix transformed to spherical coordinates, where the first row is azimuth, the second row is elevation, and the third row is range
  """
  r = np.sqrt(x**2 + y**2 + z**2)
  s = np.sqrt(x**2 + y**2)
  # Rows of rotation matrix are (az, el, r), respecively
  # See https://robotics.stackexchange.com/questions/2556/how-to-rotate-covariance for converting covariance matrices to new coordinate systems
  R = np.array([[-y/s**2, x/s**2, 0],
                [x*z/(r**2*s), y*z/(r**2*s), -s/r**2],
                [x/r, y/r, z/r]])
  return R @ cart_covar @ R.T
