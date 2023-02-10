import numpy as np
from typing import List
import scipy.constants as sc

from mpar_sim.beam.common import beamwidth2aperture
from mpar_sim.looks.look import Look


class ResourceManager():
  """The base resource manager class to manage phased array resources"""

  def allocate(self, look: Look) -> bool:
    """
    Allocate resources for the current look

    Parameters
    ----------
    look : Look
        Input look to allocate resources for

    Returns
    -------
    bool
        True if resources could be allocated, false otherwise
    """
    return True


class PowerApertureManager(ResourceManager):
  def __init__(self,
               max_average_tx_power: float = np.Inf,
               full_aperture_size: List[float] = np.Inf,
               ) -> None:
    self.max_average_tx_power = max_average_tx_power
    self.full_aperture_size = full_aperture_size

    self.total_pap = max_average_tx_power * np.prod(full_aperture_size)
    self.available_pap = self.total_pap

  def allocate(self, look: Look) -> bool:
    beamwidths = np.array([look.azimuth_beamwidth, look.elevation_beamwidth])
    wavelength = sc.c / look.center_frequency
    Lx, Ly = beamwidth2aperture(beamwidths, wavelength)

    # Compute the average tx power of the look
    look_pap = self.compute_pap(
        look.tx_power, look.pulsewidth, look.prf, [Lx, Ly])

    if look_pap <= self.available_pap:
      self.available_pap -= look_pap
      return True
    else:
      return False

  def reset(self) -> None:
    """
    Reset the resource manager to its initial state
    """
    self.available_pap = self.total_pap

  @staticmethod
  def compute_pap(tx_power: float,
                  pulsewidth: float,
                  prf: float,
                  aperture_size: List[float]):
    return tx_power * pulsewidth * prf * np.prod(aperture_size)
