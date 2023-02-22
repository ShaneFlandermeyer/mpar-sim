import datetime
from typing import Union

import numpy as np

from mpar_sim.common.matrix import block_diag
from mpar_sim.models.transition.base import TransitionModel


class LinearTransitionModel(TransitionModel):
  """Base class for linear transition models."""


class ConstantVelocity(LinearTransitionModel):
  r"""This is a class implementation of a discrete, time-variant 1D
    Linear-Gaussian Constant Velocity Transition Model.

    The target is assumed to move with (nearly) constant velocity, where
    target acceleration is modelled as white noise.

    This is a faster than the :class:`ConstantVelocity` model in StoneSoup since it can be vectorized and avoids the for loops in the transition and covariance matrix computations. This also makes it less extensible to higher-order motion models.

    The model is described by the following SDEs:

        .. math::
            :nowrap:

            \begin{eqnarray}
                dx_{pos} & = & x_{vel} d & | {Position \ on \
                X-axis (m)} \\
                dx_{vel} & = & q\cdot dW_t,\ W_t \sim \mathcal{N}(0,q^2) & | \
                Speed on\ X-axis (m/s)
            \end{eqnarray}

    Or equivalently:

        .. math::
            x_t = F_t x_{t-1} + w_t,\ w_t \sim \mathcal{N}(0,Q_t)

    where:

        .. math::
            x & = & \begin{bmatrix}
                        x_{pos} \\
                        x_{vel}
                \end{bmatrix}

        .. math::
            F_t & = & \begin{bmatrix}
                        1 & dt\\
                        0 & 1
                \end{bmatrix}

        .. math::
            Q_t & = & \begin{bmatrix}
                        \frac{dt^3}{3} & \frac{dt^2}{2} \\
                        \frac{dt^2}{2} & dt
                \end{bmatrix} q
    """

  def __init__(self, ndim_pos=3, noise_diff_coeff=1):
    self.ndim_pos = ndim_pos
    self.noise_diff_coeff = noise_diff_coeff

    self.ndim_state = self.ndim_pos*2
    self.ndim = self.ndim_state

  def function(self,
               state: np.ndarray,
               time_interval: Union[float, datetime.timedelta] = 0,
               noise: Union[bool, np.ndarray] = False,
               ) -> np.ndarray:
    if noise:
      num_samples = state.shape[1] if state.ndim > 1 else 1
      noise = self.sample_noise(num_samples=num_samples, time_interval=time_interval)
      noise = noise.reshape(state.shape)
    else:
      noise = 0

    return np.dot(self.matrix(time_interval), state) + noise

  def matrix(self, time_interval: Union[float, datetime.timedelta]):
    if isinstance(time_interval, datetime.timedelta):
      dt = time_interval.total_seconds()
    else:
      dt = time_interval
    F = np.array([[1, dt],
                  [0, 1]])
    F = block_diag(F, nreps=self.ndim_pos)
    return F

  def covar(self, time_interval: datetime.timedelta):
    if isinstance(time_interval, datetime.timedelta):
      dt = time_interval.total_seconds()
    else:
      dt = time_interval
    # TODO: Extend this to handle different noise_diff_coeff for each dimension
    covar = np.array([[dt**3/3, dt**2/2],
                      [dt**2/2, dt]]) * self.noise_diff_coeff
    covar = block_diag(covar, nreps=self.ndim_pos)
    return covar

  def sample_noise(self,
          num_samples: int = 1,
          time_interval: datetime.timedelta = 0) -> np.ndarray:
    covar = self.covar(time_interval)
    noise = np.random.multivariate_normal(
        np.zeros(self.ndim), covar, num_samples)
    return np.atleast_2d(noise).T