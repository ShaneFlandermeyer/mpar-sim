import datetime
from typing import Union
import numpy as np
from stonesoup.types.state import State, StateVector, StateVectors
from stonesoup.models.transition.linear import LinearGaussianTransitionModel
from stonesoup.models.base import Property
from stonesoup.types.array import CovarianceMatrix

from mpar_sim.common.matrix import block_diag


class ConstantVelocity(LinearGaussianTransitionModel):
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
  ndim_pos: int = Property(
      doc="The number of position dimensions of the model",
      default=3)
  noise_diff_coeff: Union[float, np.ndarray] = Property(
      doc="The velocity noise diffusion coefficient :math:`q`")

  @property
  def ndim_state(self):
    return self.ndim_pos*2

  def function(self,
               state: Union[State, StateVector, StateVectors],
               noise: Union[bool, np.ndarray] = False,
               **kwargs) -> Union[StateVector, StateVectors]:
    if isinstance(state, State):
      state = state.state_vector

    if noise:
      noise = self.rvs(num_samples=state.shape[1], **kwargs)
    else:
      noise = 0

    return self.matrix(**kwargs) @ state + noise

  def matrix(self, time_interval: datetime.timedelta, **kwargs):
    dt = time_interval.total_seconds()
    F = np.array([[1, dt],
                  [0, 1]])
    F = block_diag(F, nreps=self.ndim_pos)
    return F

  def covar(self, time_interval, **kwargs):
    dt = time_interval.total_seconds()
    # TODO: Extend this to handle different noise_diff_coeff for each dimension
    covar = np.array([[dt**3/3, dt**2/2],
                      [dt**2/2, dt]]) * self.noise_diff_coeff
    covar = block_diag(covar, nreps=self.ndim_pos)
    return CovarianceMatrix(covar)
