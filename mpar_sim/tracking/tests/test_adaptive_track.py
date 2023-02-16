import numpy as np
import pytest
from mpar_sim.models.transition.linear import ConstantVelocity
from mpar_sim.tracking.adaptive_track import adaptive_revisit_interval
from mpar_sim.tracking.kalman import kalman_predict


def test_adaptive_update_interval():
  state_vector = 100*np.ones((6,))
  covar = np.diag(1*np.ones((6,)))
  transition_model = ConstantVelocity()

  track_sharpness = 0.15
  min_revisit_interval = 0.5
  max_revisit_interval = 3.5

  # tracker = AdaptiveTracker(predict_func=lambda x, P: transition_matrix @ x)
  dt = adaptive_revisit_interval(state_vector,
                                 covar,
                                 predict_func=kalman_predict,
                                 transition_model=transition_model,
                                 beamwidths=[3, 3],
                                 track_sharpness=track_sharpness,
                                 min_revisit_interval=min_revisit_interval,
                                 max_revisit_interval=max_revisit_interval,)
  assert dt == 1.0



if __name__ == '__main__':
  pytest.main()
