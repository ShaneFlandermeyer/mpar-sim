import numpy as np

from mpar_sim.common.wrap_to_interval import wrap_to_interval


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

  def update_position(self):
    self.velocity = np.where(
        np.logical_or(
            (self.position + self.velocity) < self.position_bounds[0],
            (self.position + self.velocity) > self.position_bounds[1]),
        -self.velocity, self.velocity)
    self.position += self.velocity

  def reset(self):
    self.position = np.random.uniform(
        self.position_bounds[0], self.position_bounds[1], size=(self.n_particles, self.n_dimensions))
    self.velocity = np.random.uniform(
        self.velocity_bounds[0], self.velocity_bounds[1], size=(self.n_particles, self.n_dimensions))

  def gaussian_mutation(self, alpha=0.25):
    """
    Perform a Gaussian mutation on all particles

    Parameters
    ----------
    alpha : float, optional
        The mutation scaling factor, which dictates the fraction of the search space used in the standard deviation computation. By default 0.25
    """
    mutate_inds = np.random.uniform(
        0, 1, size=self.n_particles) < self.mutation_rate
    if mutate_inds.any():
      sigma = alpha*(self.position_bounds[1] -
                     self.position_bounds[0]).reshape(1, -1)
      sigma = np.repeat(sigma, np.count_nonzero(mutate_inds), axis=0)

      self.position[mutate_inds] += np.random.normal(
          np.zeros_like(sigma), sigma)

      # If the particle position exceeds the az/el bounds, wrap it back into the range on the other side. This improves diversity when detections are at the edge of the space compared to simple clipping.
      self.position[mutate_inds] = wrap_to_interval(
          self.position[mutate_inds], self.position_bounds[0], self.position_bounds[1])
