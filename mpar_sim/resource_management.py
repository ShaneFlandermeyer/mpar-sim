from collections import deque
from mpar_sim.beam.beam import GaussianBeam
from mpar_sim.radar import PhasedArrayRadar
from mpar_sim.looks.look import Look, Look
from mpar_sim.beam.common import beamwidth2aperture
import numpy as np
import datetime
from datetime import timedelta
import warnings

class ResourceManager():
  """
  Abstract resource managemer class
  """

  def allocate(self, look: Look) -> bool:
    """
    Allocate resources to a look request
    """
    raise NotImplementedError


class PAPResourceManager():
  """
  Manages resource constraints for a multi-function phased array by ensuring that power-aperture product (PAP) limits are not exceeded during resource allocation.   
  """

  def __init__(self,
               radar: PhasedArrayRadar,
               max_duty_cycle: float = 0.1,
               max_bandwidth: float = 100e6,
               ):
    self.radar = radar
    self.max_duty_cycle = max_duty_cycle
    self.max_bandwidth = max_bandwidth

    # Compute the maximum average power that can be transmitted by the radar
    peak_tx_power = radar.n_elements_x * \
        radar.n_elements_y*radar.element_tx_power
    max_average_tx_power = peak_tx_power * max_duty_cycle

    # Compute the aperture area of the entire array
    eff_aperture = (radar.n_elements_x * radar.element_spacing * radar.wavelength) * \
        (radar.n_elements_y * radar.element_spacing * radar.wavelength)

    # Compute the maximum power-aperture product that can be allocated
    self.total_pap = max_average_tx_power * eff_aperture

    self.allocated_tasks = []
    
  @property
  def available_pap(self):
    return self.total_pap - \
        np.sum([task.power_aperture_product for task in self.allocated_tasks])

  def allocate(self, 
               look: Look,
               current_time: datetime.datetime) -> bool:
    """
    Attempt to allocate resources for the given look.

    Parameters
    ----------
    look : Look
        The radar look to allocate

    Returns
    -------
    bool
        True if the look was successfully be allocated, False otherwise
    """

    # Compute the required aperture for the requested beamwidth
    beamwidth = np.array([look.azimuth_beamwidth, look.elevation_beamwidth])
    x_length, y_length = beamwidth2aperture(
        beamwidth, self.radar.wavelength)
    required_aperture = x_length * y_length

    # Compute the number of elements needed to achieve the given aperture size and the average power for the look
    n_elements_x = np.ceil(
        x_length / (self.radar.element_spacing * self.radar.wavelength))
    n_elements_y = np.ceil(
        y_length / (self.radar.element_spacing * self.radar.wavelength))
    peak_tx_power = n_elements_x * n_elements_y * self.radar.element_tx_power
    look.tx_power = peak_tx_power
    
    average_tx_power = peak_tx_power * look.pulsewidth * look.prf

    # Schedule the look if resource constraints are met
    look_pap = average_tx_power * required_aperture
    
    if look_pap > self.total_pap:
      # Warn the user
      warnings.warn("Look PAP exceeds maximum PAP of the radar. Look cannot be scheduled.")

    if look_pap <= self.available_pap and look.bandwidth <= self.max_bandwidth:
      look.power_aperture_product = look_pap
      look.start_time = current_time
      self.allocated_tasks.append(look)
      return True
    else:
      return False