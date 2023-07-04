from mpar_sim.types.target import Target
import pytest
import jax.numpy as jnp
from mpar_sim.models.rcs import Swerling
from mpar_sim.models.transition.linear import ConstantVelocity

if __name__ == '__main__':
  t = Target(
    position=jnp.array([0, 0, 0]),
    velocity=jnp.array([1, 1, 1]),
    transition_model=ConstantVelocity(ndim_pos=3, noise_diff_coeff=1),
    rcs=Swerling(case=1, mean=100),
  )
  t.move(dt=1, noise=True)
  print(t.position)
  t.move(dt=1, noise=True)
  print(t.position)
  # print(t.detection_probability(pfa=1e-8, n_pulse=10, snr_db=
  # 
  # 10))