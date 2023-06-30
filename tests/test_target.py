from mpar_sim.types.target import Target
import pytest
import jax.numpy as jnp
from mpar_sim.models.rcs import Swerling
from pprint import pprint

# @pytest.fixture()
class TestTarget():
  def test_rcs():
    pass

if __name__ == '__main__':
  t = Target(
    rcs_model=Swerling(case=3, mean=100)
  )
  print(t.rcs)
  # print(t.detection_probability(pfa=1e-8, n_pulse=10, snr_db=10))