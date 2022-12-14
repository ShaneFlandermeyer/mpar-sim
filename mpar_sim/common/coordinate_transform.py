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
    r: Union[float, np.ndarray]
) -> Union[Tuple[float, float, float], Tuple[np.ndarray, np.ndarray, np.ndarray]]:
  """
  Convert spherical coordinates to cartesian

  Args:
      azimuth (Union[float, np.ndarray]): Azimuth angle in radians
      elevation (Union[float, np.ndarray]): Elevation angle in radians
      r (Union[float, np.ndarray]): Range

  Returns:
      Union[float, np.ndarray]: Cartesian coordinates
  """
  x = r * np.cos(elevation) * np.cos(azimuth)
  y = r * np.cos(elevation) * np.sin(azimuth)
  z = r * np.sin(elevation)

  return (x, y, z)


def cart2sph(
    x: Union[float, np.ndarray],
    y: Union[float, np.ndarray],
    z: Union[float, np.ndarray]
) -> Union[Tuple[float, float, float], Tuple[np.ndarray, np.ndarray, np.ndarray]]:
  """
  Convert cartesian coordinates to spherical

  Args:
      x (Union[float, np.ndarray]): X coordinate
      y (Union[float, np.ndarray]): Y coordinate
      z (Union[float, np.ndarray]): Z coordinate

  Returns:
      Union[float, np.ndarray]: Spherical coordinates
  """
  azimuth = np.arctan2(y, x)
  elevation = np.arctan2(z, np.sqrt(x**2 + y**2))
  r = np.sqrt(x**2 + y**2 + z**2)

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
