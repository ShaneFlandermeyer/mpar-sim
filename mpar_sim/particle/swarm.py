import numpy as np


class ParticleSwarm():
  def __init__(self,
               n_particles: int,
               n_dimensions: int,
               position_bounds: np.ndarray,
               velocity_bounds: np.ndarray):
    self.n_particles = n_particles
    self.n_dimensions = n_dimensions
    self.position_bounds = position_bounds
    self.velocity_bounds = velocity_bounds
    self.reset()

  def reset(self):
    self.position = np.random.uniform(
        self.position_bounds[0], self.position_bounds[1], size=(self.n_particles, self.n_dimensions))
    self.velocity = np.random.uniform(
        self.velocity_bounds[0], self.velocity_bounds[1], size=(self.n_particles, self.n_dimensions))
