import pytest
from mpar_sim.resource.manager import ResourceManager
from mpar_sim.resource.scheduler import BestFirstScheduler
from mpar_sim.types.look import Look
import numpy as np


class TestBestFirstScheduler:
  @pytest.fixture
  def scheduler(self) -> BestFirstScheduler:
    return BestFirstScheduler()

  @pytest.fixture
  def manager(self) -> ResourceManager:
    return ResourceManager()

  def test_run(self, scheduler: BestFirstScheduler, manager: ResourceManager):
    """
    Tests that the scheduler runs successfully
    """
    looks = [Look()]*5
    scheduled_looks, deferred_looks = scheduler.schedule(
        looks, manager, None)

    assert len(scheduled_looks) == 5
    assert len(deferred_looks) == 0

  def test_priority(self, scheduler: BestFirstScheduler, manager: ResourceManager):
    """
    Tests that higher-priority tasks are not delayed by lower-priority tasks
    """
    scheduler.sort_key = "start_time"
    priorities = np.arange(5)
    start_times = np.zeros(5)
    looks = [Look(priority=p, start_time=t)
             for p, t in zip(priorities, start_times)]
    scheduled_looks, deferred_looks = scheduler.schedule(
        looks, manager, None)
    assert len(scheduled_looks) == 5
    assert len(deferred_looks) == 0
    assert np.all(
        [l.priority == p for l, p in zip(scheduled_looks, np.flip(priorities))])


if __name__ == '__main__':
  pytest.main()
