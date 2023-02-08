from mpar_sim.old.radar_env import RadarEnvironment
from mpar_sim.old.radar_detection_generator import RadarDetectionGenerator
import numpy as np
from mpar_sim.old.platforms import Platform
import time
from mpar_sim.beam.beam import Beam, RectangularBeam, GaussianBeam
from mpar_sim.common.coordinate_transform import sph2cart
from mpar_sim.radar import Radar
from mpar_sim.agents import RasterScanAgent
import matplotlib.pyplot as plt

if __name__ == '__main__':
  start = time.time()

  # Create targets
  target_az = 0
  target_el = 0
  target_range = 10e3
  x, y, z = sph2cart(target_az, target_el, target_range)

  targets = []
  targets = [Platform(position=np.array([x, y, z], dtype=np.float32),
                      velocity=np.array([1, 2, 3], dtype=np.float32), rcs=10) for _ in range(3)]
  targets += [Platform(position=np.array([x+200, y+1000, z+200], dtype=np.float32),
                       velocity=np.array([1, 2, 3], dtype=np.float32), rcs=10)]

  # Create radar object
  radar = Radar(
      # Array parameters
      n_elements_x=16,
      n_element_y=16,
      element_spacing=0.5,
      element_tx_power=10,
      # System parameters
      center_frequency=3e9,
      system_temperature=290,
      noise_figure=4,
      # Scan settings
      beam_type=RectangularBeam,
      azimuth_limits=np.array([-60, 60]),
      elevation_limits=np.array([-20, 20]),
      # Detection settings
      false_alarm_rate=1e-6,
  )

  # Create a search agent that attempts to find targets through a raster scan throughout the entire volume
  raster_agent = RasterScanAgent(
      azimuth_scan_limits=radar.azimuth_limits,
      elevation_scan_limits=radar.elevation_limits,
      azimuth_beam_spacing=0.85,
      elevation_beam_spacing=0.85,
      azimuth_beamwidth=7.5,
      elevation_beamwidth=7.5,
      bandwidth=5e6,
      pulsewidth=10e-6,
      prf=1500,
      n_pulses=10)

  env = RadarEnvironment(
      targets=targets,
      radar=radar
  )

  # dt = raster_agent.n_pulses/raster_agent.prf
  observations = []
  n_timesteps = 200
  for i in range(n_timesteps):
    
    t = env.global_time
    # Select a new set of task parameters
    action = raster_agent.act(current_time=t)
    
    observation, detections = env.step(action)
    env.render()
    if len(detections) > 0:
      print(detections)
    
  print("Time elapsed:", time.time() - start)
  env.close()