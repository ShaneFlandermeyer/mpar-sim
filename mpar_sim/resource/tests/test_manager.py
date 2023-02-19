import pytest
from mpar_sim.resource.manager import PowerApertureManager
from mpar_sim.types.look import Look
import scipy.constants as sc


def test_power_aperture_manager():
  # Create a look that uses the entire array
  center_frequency = 3e9
  wavelength = sc.c / center_frequency
  nx = 32
  ny = 32
  element_tx_power = 10
  element_spacing = 0.5
  Lx = nx * element_spacing * wavelength
  Ly = ny * element_spacing * wavelength

  max_duty_cycle = 0.1
  max_tx_power = nx*ny*element_tx_power
  max_average_tx_power = max_tx_power*max_duty_cycle

  pulsewidth = 10e-6
  pri = pulsewidth / max_duty_cycle
  prf = 1 / pri
  look = Look(tx_power=max_tx_power,
              azimuth_beamwidth=3.2,
              elevation_beamwidth=3.2,
              pulsewidth=pulsewidth,
              prf=prf,
              center_frequency=center_frequency)

  # First allocation of the look should succeed
  manager = PowerApertureManager(max_average_tx_power, [Lx, Ly])
  success = manager.allocate(look)
  assert success

  # Second should fail
  success = manager.allocate(look)
  assert not success

  # Should succeed again after a reset
  manager.reset()
  success = manager.allocate(look)
  assert success

if __name__ == '__main__':
  pytest.main()
