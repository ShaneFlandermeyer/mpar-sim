import numpy as np
import pytest
from mpar_sim.models.transition.linear import ConstantVelocity
from mpar_sim.tracking.adaptive_track import AdaptiveTrackManager, adaptive_revisit_interval
from mpar_sim.tracking.kalman import kalman_predict
from mpar_sim.types.detection import Detection, TrueDetection
from mpar_sim.types.groundtruth import GroundTruthPath


class TestAdaptiveTrackManager():
  @pytest.fixture
  def manager(self):
    return AdaptiveTrackManager(
        track_sharpness=0.15,
        confirmation_interval=1/20,
        min_revisit_interval=0.2,
        max_revisit_interval=2.0,
    )

  def test_process_detections(self, manager: AdaptiveTrackManager):
    # Make a fake detection to add to the track list
    detections = [TrueDetection(GroundTruthPath())]
    time = 0
    manager.process_detections(detections, time)
    assert len(manager.tentative_tracks) == 1
    assert len(manager.confirmed_tracks) == 0

    track_id = manager.tentative_tracks[0].id
    assert manager.update_times[track_id] == time + \
        manager.confirmation_interval

    # Add another detection to test the tentative track logic
    time += manager.confirmation_interval
    manager.process_detections(detections, time)

    assert len(manager.tentative_tracks) == 1
    assert len(manager.confirmed_tracks) == 0

    assert manager.update_times[track_id] == time + \
        manager.confirmation_interval


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
                                 az_beamwidth=3,
                                 el_beamwidth=3,
                                 track_sharpness=track_sharpness,
                                 min_revisit_interval=min_revisit_interval,
                                 max_revisit_interval=max_revisit_interval,)
  assert dt == 1.0


if __name__ == '__main__':
  pytest.main()
