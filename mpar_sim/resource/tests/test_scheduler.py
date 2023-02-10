import pytest
from mpar_sim.resource.manager import ResourceManager
from mpar_sim.resource.scheduler import BestFirstScheduler
from mpar_sim.looks.look import Look

def test_schedule():
  looks = [Look()]*5
  scheduler = BestFirstScheduler()
  manager = ResourceManager()
  scheduled_looks, deferred_looks = scheduler.schedule(looks, manager, None)

  assert len(scheduled_looks) == 5
  assert len(deferred_looks) == 0

if __name__ == '__main__':
  pytest.main()
