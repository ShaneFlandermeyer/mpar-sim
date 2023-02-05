import numpy as np
from mpar_sim.particle.swarm import ParticleSwarm


class SurveillanceSwarm(ParticleSwarm):
  def __init__(self,
               gravity: float,
               min_dispersion_inertia: float,
               max_dispersion_inertia: float,
               detection_inertia: float,
               adaptive_inertia_rate: float = 1.5,
               *args,
               **kwargs):
    super().__init__(*args, **kwargs)
    self.gravity = gravity
    self.min_dispersion_inertia = min_dispersion_inertia
    self.max_dispersion_inertia = max_dispersion_inertia
    self.adaptive_inertia_rate = adaptive_inertia_rate
    self.inertia = np.ones(
        (self.n_particles, 1))*self.max_dispersion_inertia
    self.range = np.zeros((self.n_particles, 1))
    self.detection_inertia = detection_inertia

  def dispersion_phase(self,
                       steering_angles: np.ndarray,
                       beamwidths: np.ndarray):
    min_angles = steering_angles - beamwidths/2
    max_angles = steering_angles + beamwidths/2
    in_beam = np.logical_and(
        np.logical_and(
            self.position[:, 0] >= min_angles[0],
            self.position[:, 0] <= max_angles[0]),
        np.logical_and(
            self.position[:, 1] >= min_angles[1],
            self.position[:, 1] <= max_angles[1]))
    # Set the velocity of each affected particle radially away from the beam
    relative_pos = self.position[in_beam] - steering_angles
    distance = np.linalg.norm(relative_pos, axis=1)[:, np.newaxis]
    radial_velocities = relative_pos / distance
    self.velocity[in_beam] = radial_velocities * \
        np.random.uniform(np.array([0.0, 0.0]),
                          beamwidths, size=radial_velocities.shape)
    # Adaptively update the inertia
    self.inertia[in_beam] *= self.adaptive_inertia_rate
    self.inertia[in_beam] = np.clip(
        self.inertia[in_beam], 0, self.max_dispersion_inertia)
    # Update the swarm and return
    self.update_position()
    self.velocity *= self.inertia
    # When a new beam is transmitted, set the magnitude of the detection velocity proportionally
    self.detection_velocity_scale = beamwidths/2

  def detection_phase(self, az: float, el: float, rng: float):
    relative_pos = np.array([[az, el]]) - self.position
    distance = np.linalg.norm(relative_pos, axis=1)[:, np.newaxis]
    move_probability = np.exp(-self.gravity*distance).ravel()
    move_inds = np.random.uniform(0, 1, size=distance.size) < move_probability
    velocity = relative_pos / distance
    self.velocity[move_inds] *= self.detection_inertia
    self.velocity[move_inds] += velocity[move_inds] * np.random.uniform(
        0, self.detection_velocity_scale, size=velocity[move_inds].shape)
    self.inertia[move_inds] = self.min_dispersion_inertia
    self.range[move_inds] = rng
    self.update_position()
