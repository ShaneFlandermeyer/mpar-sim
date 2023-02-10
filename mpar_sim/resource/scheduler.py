import numpy as np
from typing import List, Tuple
from datetime import datetime
import operator

from mpar_sim.looks.look import Look
from mpar_sim.resource_management import ResourceManager


class BestFirstScheduler():

  def __init__(self,
               sort_key: str = "start_time",
               reverse: bool = False) -> None:
    self.sort_key = sort_key
    self.reverse = reverse

  def schedule(self,
               requests: List[Look],
               manager: ResourceManager,
               current_time: datetime) -> Tuple[List[Look], List[Look]]:
    """Attempt to schedule a list of looks using a best-first strategy.

    Parameters
    ----------
    requests : List[Look]
        A list of requests for access to phased array resources
    manager: ResourceManager
        The resource manager object that is used to allocate resources
    current_time : datetime
        The current time

    Returns
    -------
    Tuple[List[Look], List[Look]]
        A tuple that contains:
         - A list of looks that were scheduled successfully
         - A list of looks that were deferred
    """
    if not isinstance(requests, list):
      requests = [requests]

    requests.sort(key=operator.attrgetter(self.sort_key), reverse=self.reverse)

    # Attempt to schedule each request. By default, it is assumed that no requests are successfully scheduled. If a request is successfully scheduled, it is added to the list of scheduled requests and removed from the list of deferred requests.
    scheduled_requests = []
    deferred_requests = requests.copy()
    for request in requests:
      success = manager.allocate(request)
      if success:
        scheduled_requests.append(request)
        deferred_requests.remove(request)
    return scheduled_requests, deferred_requests
