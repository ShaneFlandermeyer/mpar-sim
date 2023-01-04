"""
These functions give valid instances of several objects implemented in this project for convenient access. The parameters are chosen to be reasonable for a typical scenario, although they're almost certainly sub-optimal
"""

from datetime import timedelta

import gymnasium as gym
import numpy as np

from mpar_sim.agents.raster_scan import RasterScanAgent
from mpar_sim.beam.beam import RectangularBeam
from mpar_sim.particle.global_best import IncrementalGlobalBestPSO
from mpar_sim.particle.local_best import IncrementalLocalBestPSO
from mpar_sim.radar import PhasedArrayRadar
from mpar_sim.resource_management import PAPResourceManager
from mpar_sim.schedulers import BestFirstScheduler


def default_radar():
  return PhasedArrayRadar(
      ndim_state=6,
      position_mapping=(0, 2, 4),
      velocity_mapping=(1, 3, 5),
      position=np.array([[0], [0], [0]]),
      rotation_offset=np.array([[0], [0], [0]]),
      # Array parameters
      n_elements_x=32,
      n_elements_y=32,
      element_spacing=0.5,  # Wavelengths
      element_tx_power=10,
      # System parameters
      center_frequency=3e9,
      system_temperature=290,
      noise_figure=4,
      # Scan settings
      beam_shape=RectangularBeam,
      az_fov=[-45, 45],
      el_fov=[-45, 45],
      # Detection settings
      false_alarm_rate=1e-6,
      include_false_alarms=False
  )


def default_raster_scan_agent():
  return RasterScanAgent(
      azimuth_scan_limits=np.array([-45, 45]),
      elevation_scan_limits=np.array([-45, 45]),
      azimuth_beam_spacing=0.75,
      elevation_beam_spacing=0.75,
      azimuth_beamwidth=5,
      elevation_beamwidth=5,
      bandwidth=100e6,
      pulsewidth=10e-6,
      prf=5e3,
      n_pulses=100,
  )


def default_gbest_pso():
  options = {'c1': 0, 'c2': 0.9, 'w': 0.3}
  return IncrementalGlobalBestPSO(n_particles=2500,
                                  dimensions=2,
                                  options=options,
                                  bounds=np.array([[-45, -45], [45, 45]]),
                                  pbest_reset_interval=1000,
                                  )


def default_lbest_pso():
  # options = {'c1': 0.2, 'c2': 0.5, 'w': 0.8}
  options = {'c1': 0.2, 'c2': 0.6, 'w': 0.5, 'k': 50, 'p': 2}
  return IncrementalLocalBestPSO(n_particles=1500,
                                 dimensions=2,
                                 options=options,
                                 bounds=np.array([[-45, -45], [45, 45]]),
                                 pbest_reset_interval=1000,
                                 static=True,
                                 )


def default_scheduler(radar: PhasedArrayRadar):
  manager = PAPResourceManager(radar,
                               max_duty_cycle=0.1,
                               max_bandwidth=100e6)
  return BestFirstScheduler(manager,
                            sort_key="start_time",
                            reverse_sort=True,
                            max_queue_size=10,
                            max_time_delta=timedelta(seconds=0.5))
