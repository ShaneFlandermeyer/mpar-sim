import numpy as np
import pytest
from mpar_sim.models.measurement.nonlinear import CartesianToRangeAzElRangeRate

from mpar_sim.models.transition.linear import ConstantVelocity
from mpar_sim.tracking.adaptive_track import (AdaptiveTrackManager,
                                              adaptive_revisit_interval)
from mpar_sim.tracking.extended_kalman import extended_kalman_predict, extended_kalman_update
from mpar_sim.tracking.kalman import kalman_predict, kalman_update
from mpar_sim.tracking.tracker import Tracker
from mpar_sim.types.detection import Detection, TrueDetection
from mpar_sim.types.groundtruth import GroundTruthPath, GroundTruthState
from mpar_sim.types.look import Look
from mpar_sim.types.state import State


class TestAdaptiveTrackManager():
  @pytest.fixture
  def manager(self):
    mm = CartesianToRangeAzElRangeRate(
        noise_covar=np.diag([0.2, 0.2, 1, 1])
    )
    tracker = Tracker(predict_func=kalman_predict,
                      update_func=extended_kalman_update,
                      transition_model=ConstantVelocity(),
                      measurement_model=mm,
                      )
    return AdaptiveTrackManager(
        track_sharpness=0.15,
        confirmation_interval=1/20,
        min_revisit_interval=0.2,
        max_revisit_interval=2.0,
        tracker=tracker,
        n_confirm_detections=3,
    )

  def test_process_detections(self, manager: AdaptiveTrackManager):
    # Make a fake detection to add to the track list
    target_path = GroundTruthPath(
        GroundTruthState(np.array([1e3, 0, 1e3, 0, 0, 0])))
    state_vector = manager.tracker.measurement_model.function(
        target_path.state_vector, noise=False)
    detections = [TrueDetection(state_vector=state_vector,
                                groundtruth_path=target_path)]

    look = Look()

    # Test the handling of the initiation and tentative tracks
    time = 0
    for _ in range(2):
      manager.process_detections(detections, time, look)
      assert len(manager.tentative_tracks) == 1
      assert len(manager.confirmed_tracks) == 0

      track_id = manager.tentative_tracks[0].id
      assert manager.update_times[track_id] == time + \
          manager.confirmation_interval

      time += manager.confirmation_interval

    # Test behavior for confirmed tracks
    manager.process_detections(detections, time, look)
    assert len(manager.tentative_tracks) == 0
    assert len(manager.confirmed_tracks) == 1

    track_id = manager.confirmed_tracks[0].id
    assert manager.update_times[track_id] == time + \
        manager.confirmation_interval

  def test_generate_looks(self, manager: AdaptiveTrackManager):
    # Make a fake detection to add to the track list
    time = 0
    target_path = GroundTruthPath(
        GroundTruthState(np.array([1e3, 0, 1e3, 0, 0, 0])))
    state_vector = manager.tracker.measurement_model.function(
        target_path.state_vector, noise=False)
    detections = [TrueDetection(state_vector=state_vector,
                                groundtruth_path=target_path,
                                timestamp=time)]
    look = Look(azimuth_beamwidth=10, elevation_beamwidth=10)

    # Add a detection to the tentative tracks list and generate an updated look on it.
    manager.process_detections(detections, time, look)
    time += manager.confirmation_interval
    looks = manager.generate_looks(time)
    
    assert len(looks) == 1
    assert looks[0].azimuth_steering_angle == 45
    assert looks[0].elevation_steering_angle == 0
    assert looks[0].start_time == manager.confirmation_interval


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
