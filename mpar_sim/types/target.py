import jax.numpy as jnp
from mpar_sim.models.transition.base import TransitionModel
from mpar_sim.models.rcs import RCSModel, Swerling

class Target():
  def __init__(
    self,
    position: jnp.array = None,
    velocity: jnp.array = None,
    transition_model: TransitionModel = None,
    rcs_model: RCSModel = None,
  ) -> None:
    self.position = position
    self.velocity = velocity
    self.transition_model = transition_model
    self.rcs_model = rcs_model
    if isinstance(self.rcs_model, float):
      self.rcs_model = Swerling(order=0, mean_rcs=self.rcs_model)
    
  @property
  def rcs(self, **kwargs) -> jnp.array:
    return self.rcs_model(**kwargs)
  
  def detection_probability(self, pfa, n_pulse, snr_db) -> float:
    return self.rcs_model.detection_probability(pfa=pfa, n_pulse=n_pulse, snr_db=snr_db)