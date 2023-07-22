from mpar_sim.types.target import Target
import pytest
from mpar_sim.models.rcs import Swerling
from mpar_sim.models.transition.constant_velocity import ConstantVelocity
import numpy as np

if __name__ == '__main__':
  t = Target(
    position=np.array([0, 0, 0]),
    velocity=np.array([1, 1, 1]),
    transition_model=ConstantVelocity(ndim_pos=3, q=1),
    rcs=Swerling(case=1, mean=100),
  )
  t.move(dt=1, noise=False)
  print(t.position)
  t.move(dt=1, noise=False)
  print(t.position)
  print(t.rcs)
  # print(t.detection_probability(pfa=1e-8, n_pulse=10, snr_db=
  # 
  # 10))