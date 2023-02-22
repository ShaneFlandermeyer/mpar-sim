#
# Author: Shane Flandermeyer
# Created on Thu Feb 16 2023
# Copyright (c) 2023
#
# Contains classes for scheduling access requests to radar resources.
#


import operator
from datetime import datetime
from typing import List, Tuple

import numpy as np

from mpar_sim.resource.manager import ResourceManager
from mpar_sim.types.look import Look


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
    while len(deferred_requests) > 0:
      
      # This scheduler is configured so that the current request cannot delay a higher-priority request in the queue. Therefore, we need to find the highest priority request and try to schedule it.
      current_request = deferred_requests[0]
      highest_priority_request = current_request
      for next_request in deferred_requests:
        if next_request.priority > highest_priority_request.priority:
          highest_priority_request = next_request
          
      success = manager.allocate(highest_priority_request)
      if success:
        scheduled_requests.append(highest_priority_request)
        deferred_requests.remove(highest_priority_request)
      else:
        break
      
    return scheduled_requests, deferred_requests