import numpy as np
from typing import Union


def wrap_to_interval(x: Union[float, np.ndarray],
                     start: float,
                     end: float) -> Union[float, np.ndarray]:
  """
  Wrap x to the interval [a, b)

  Args:
      x (Union[float, np.ndarray]): Unwrapped input value
      start (float): Start of interval
      end (float): End of interval

  Returns:
      Union[float, np.ndarray]: _description_
  """
  return (x - start) % (end - start) + start