from typing import List, Union
from mpar_sim.looks.look import Look
from mpar_sim.resource_management import ResourceManager
from collections import deque
import operator
import bisect
import datetime


class BestFirstScheduler():
  """
  Scheduler that sorts the task list based on one of the task attributes (in decreasing order) then greedily schedules the tasks in that order.
  """

  def __init__(self,
               resource_manager: ResourceManager,
               sort_key: str = "priority",
               reverse_sort: bool = False,
               max_queue_size: int = None,
               max_time_delta: datetime.timedelta = None,
               ) -> None:
    self.manager = resource_manager
    self.sort_key = sort_key
    self.max_queue_size = max_queue_size
    self.reverse_sort = reverse_sort
    self.max_time_delta = max_time_delta
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

    # Sort the task queue in ascending order by the desired attribute
    self.task_list.sort(key=operator.attrgetter(
        self.sort_key), reverse=self.reverse_sort)

    # Limit the maximum size of the queue. If the operation above exceeds the queue, remove the tasks that are least important with respect to the parameters we are sorting by
    # if self.max_queue_size is not None and len(self.task_list) > self.max_queue_size:
    #   self.task_list = self.task_list[:self.max_queue_size]

  def schedule(self, look_requests: List[Look], current_time: datetime.datetime) -> List[Look]:
    # Add look requests to the scheduler queue
    self.append(look_requests)

    # Allocate resources greedily until there are no more resources available
    for task in self.task_list.copy():
      if task.start_time < current_time - self.max_time_delta:
        self.task_list.remove(task)
        continue
      elif task.start_time > current_time + self.max_time_delta:
        continue

      # Try to allocate resources for the current task
      allocation_success = self.manager.allocate(task, current_time)

      if allocation_success:
        # If the task was successfully allocated, schedule it for execution and remove it from the internal task list.
        self.task_list.remove(task)
      else:
        break