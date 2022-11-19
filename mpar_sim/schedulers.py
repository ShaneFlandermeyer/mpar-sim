from typing import List, Union
from mpar_sim.look import Look
from mpar_sim.resource_management import ResourceManager
from collections import deque
import operator
import bisect
import datetime


class PriorityScheduler():
  """
  Scheduler that uses priority to determine the order in which look requests are executed.
  """

  def __init__(self,
               max_queue_size=None,
               sort_key: str = "priority"
               ) -> None:
    self.max_queue_size = max_queue_size
    self.task_list = []

  def append(self, look_requests: Union[Look, List[Look]]):
    """
    Add look requests to the scheduler queue and sort the queue by priority

    Parameters
    ----------
    look_requests : Union[Look, List[Look]]
        Look requests to schedule
    """
    # Add to the list of tasks
    if isinstance(look_requests, Look):
      self.task_list.append(look_requests)
    else:
      self.task_list.extend(look_requests)

    # Limit the maximum size of the queue. If the operation above exceeds the queue
    if self.max_queue_size is not None and len(self.task_list) > self.max_queue_size:
      self.task_list = self.task_list[-self.max_queue_size:]

    # Sort the task queue in ascending order by the desired attribute
    # TODO: Make the attribute to sort by an object parameter
    self.task_list.sort(key=operator.attrgetter("priority"), reverse=False)

  def schedule(self,
               current_time: datetime.datetime,
               resource_manager: ResourceManager) -> List[Look]:
    # TODO: Check for stale tasks that have been delayed for too long to be useful

    # Allocate resources greedily until there are no more resources available
    scheduled_tasks = []
    for itask in range(len(self.task_list.copy())):
      # Try to allocate resources for the current task
      task = self.task_list[itask]
      allocation_success = resource_manager.allocate(task)

      if allocation_success:
        # If the task was successfully allocated, schedule it for execution and remove it from the internal task list.
        scheduled_tasks.append(task)
        self.task_list.pop(itask)
      else:
        # Once no more tasks can be scheduled, send the main program the tasks to execute
        return scheduled_tasks


if __name__ == '__main__':
  queue = [20, 40, 60, 80]
  # queue.extend(10)
  queue.sort(reverse=True)
  print(queue)
