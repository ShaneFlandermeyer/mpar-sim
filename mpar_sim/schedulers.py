from typing import List, Union
from mpar_sim.looks.look import Look
from mpar_sim.resource_management import ResourceManager
from collections import deque
import operator
import bisect
import datetime


class MaxGreedyScheduler():
  """
  Scheduler that sorts the task list based on one of the task attributes (in decreasing order) then greedily schedules the tasks in that order.
  """

  def __init__(self,
               resource_manager: ResourceManager,
               sort_key: str = "priority",
               max_queue_size: int = None,
               ) -> None:
    self.manager = resource_manager
    self.sort_key = sort_key
    self.max_queue_size = max_queue_size
    self.task_list = []

  def append(self, look_requests: List[Look]):
    """
    Add look requests to the scheduler queue and sort the queue by priority

    Parameters
    ----------
    look_requests : Union[Look, List[Look]]
        Look requests to schedule
    """
    if not isinstance(look_requests, list):
      look_requests = [look_requests]

    # Add to the list of tasks
    self.task_list.extend(look_requests)

    # Limit the maximum size of the queue. If the operation above exceeds the queue
    if self.max_queue_size is not None and len(self.task_list) > self.max_queue_size:
      # TODO: This should remove the oldest tasks
      self.task_list = self.task_list[-self.max_queue_size:]

    # Sort the task queue in ascending order by the desired attribute
    self.task_list.sort(key=operator.attrgetter(self.sort_key), reverse=False)

  def schedule(self, look_requests: List[Look], current_time: datetime.datetime) -> List[Look]:
    # Add look requests to the scheduler queue
    self.append(look_requests)
    
    # TODO: Check for stale tasks that have been delayed for too long to be useful

    # Allocate resources greedily until there are no more resources available
    for task in self.task_list.copy():
      # Try to allocate resources for the current task
      allocation_success = self.manager.allocate(task, current_time)

      if allocation_success:
        # If the task was successfully allocated, schedule it for execution and remove it from the internal task list.
        self.task_list.remove(task)
      else:
        break
