from typing import Tuple, Union

import numpy as np
from scipy.linalg import block_diag, inv, pinv
from scipy.stats import multivariate_normal
from stonesoup.base import Property, clearable_cached_property
from stonesoup.functions import (build_rotation_matrix, cart2angles, cart2pol,
                                 cart2sphere, pol2cart, sphere2cart)
from stonesoup.models.base import GaussianModel, LinearModel, ReversibleModel
from stonesoup.models.measurement.base import MeasurementModel
from stonesoup.models.measurement.nonlinear import NonLinearGaussianMeasurement
from stonesoup.types.angle import Bearing, Elevation
from stonesoup.types.array import CovarianceMatrix, StateVector, StateVectors
from stonesoup.types.numeric import Probability

from mpar_sim.common import wrap_to_interval


class CartesianToElevationBearingRangeRate(NonLinearGaussianMeasurement, ReversibleModel):
  r"""This is a class implementation of a time-invariant measurement model, \
  where measurements are assumed to be received in the form of elevation \
  (:math:`\theta`),  bearing (:math:`\phi`), range (:math:`r`) and
  range-rate (:math:`\dot{r}`), with Gaussian noise in each dimension.

  The model is described by the following equations:

  .. math::

    \vec{y}_t = h(\vec{x}_t, \vec{v}_t)

  where:

  * :math:`\vec{y}_t` is a measurement vector of the form:

  .. math::

    \vec{y}_t = \begin{bmatrix}
              \theta \\
              \phi \\
              r \\
              \dot{r}
          \end{bmatrix}

  * :math:`h` is a non-linear model function of the form:

  .. math::

    h(\vec{x}_t,\vec{v}_t) = \begin{bmatrix}
              asin(\mathcal{z}/\sqrt{\mathcal{x}^2 + \mathcal{y}^2 +\mathcal{z}^2}) \\
              atan2(\mathcal{y},\mathcal{x}) \\
              \sqrt{\mathcal{x}^2 + \mathcal{y}^2 + \mathcal{z}^2} \\
              (x\dot{x} + y\dot{y} + z\dot{z})/\sqrt{x^2 + y^2 + z^2}
              \end{bmatrix} + \vec{v}_t

  * :math:`\vec{v}_t` is Gaussian distributed with covariance :math:`R`, i.e.:

  .. math::

    \vec{v}_t \sim \mathcal{N}(0,R)

  .. math::

    R = \begin{bmatrix}
          \sigma_{\theta}^2 & 0 & 0 & 0\\
          0 & \sigma_{\phi}^2 & 0 & 0\\
          0 & 0 & \sigma_{r}^2 & 0\\
          0 & 0 & 0 & \sigma_{\dot{r}}^2
          \end{bmatrix}

  The :py:attr:`mapping` property of the model is a 3 element vector, \
  whose first (i.e. :py:attr:`mapping[0]`), second (i.e. \
  :py:attr:`mapping[1]`) and third (i.e. :py:attr:`mapping[2]`) elements \
  contain the state index of the :math:`x`, :math:`y` and :math:`z`  \
  coordinates, respectively.

  Note
  ----
  This class implementation assuming at 3D cartesian space, it therefore \
  expects a 6D state space.
  """

  translation_offset: StateVector = Property(
      default=None,
      doc="A 3x1 array specifying the origin offset in terms of :math:`x,y,z` coordinates.")
  velocity_mapping: Tuple[int, int, int] = Property(
      default=(1, 3, 5),
      doc="Mapping to the targets velocity within its state space")
  velocity: StateVector = Property(
      default=None,
      doc="A 3x1 array specifying the sensor velocity in terms of :math:`x,y,z` coordinates.")

  def __init__(self, *args, **kwargs):
    """
    Ensure that the translation offset is initiated
    """
    super().__init__(*args, **kwargs)
    # Set values to defaults if not provided
    if self.translation_offset is None:
      self.translation_offset = StateVector([0] * 3)

    if self.velocity is None:
      self.velocity = StateVector([0] * 3)

  @property
  def ndim_meas(self) -> int:
    """ndim_meas getter method

    Returns
    -------
    :class:`int`
        The number of measurement dimensions
    """

    return 4

  def function(self, state, noise=False, **kwargs) -> StateVector:
    r"""Model function :math:`h(\vec{x}_t,\vec{v}_t)`

    Parameters
    ----------
    state: :class:`~.StateVector`
        An input state vector for the target

    noise: :class:`numpy.ndarray` or bool
        An externally generated random process noise sample (the default is
        `False`, in which case no noise will be added
        if 'True', the output of :meth:`~.Model.rvs` is added)

    Returns
    -------
    :class:`numpy.ndarray` of shape (:py:attr:`~ndim_state`, 1)
        The model function evaluated given the provided time interval.
    """

    if isinstance(noise, bool) or noise is None:
      if noise:
        noise = self.rvs(num_samples=state.state_vector.shape[1], **kwargs)
      else:
        noise = 0

    # Account for origin offset in position to enable range and angles to be determined
    xyz_pos = state.state_vector[self.mapping, :] - self.translation_offset

    # Rotate coordinates based upon the sensor_velocity
    xyz_rot = self.rotation_matrix @ xyz_pos

    # Convert to Spherical
    rho, phi, theta = cart2sphere(xyz_rot[0, :], xyz_rot[1, :], xyz_rot[2, :])

    # Determine the net velocity component in the engagement
    xyz_vel = state.state_vector[self.velocity_mapping, :] - self.velocity

    # Use polar to calculate range rate
    rr = np.einsum('ij,ij->j', xyz_pos, xyz_vel) / \
        np.linalg.norm(xyz_pos, axis=0)

    bearings = [Bearing(i) for i in phi]
    elevations = [Elevation(i) for i in theta]
    return StateVectors([elevations,
                         bearings,
                         rho,
                         rr]) + noise

  def inverse_function(self, detection, **kwargs) -> StateVector:
    theta, phi, rho, rho_rate = detection.state_vector

    x, y, z = sphere2cart(rho, phi, theta)
    # because only rho_rate is known, only the components in
    # x,y and z of the range rate can be found.
    x_rate = np.cos(phi) * np.cos(theta) * rho_rate
    y_rate = np.cos(phi) * np.sin(theta) * rho_rate
    z_rate = np.sin(phi) * rho_rate

    inv_rotation_matrix = inv(self.rotation_matrix)

    out_vector = StateVector(np.zeros((self.ndim_state, 1)))
    out_vector[self.mapping, 0] = x, y, z
    out_vector[self.velocity_mapping, 0] = x_rate, y_rate, z_rate

    out_vector[self.mapping,
               :] = inv_rotation_matrix @ out_vector[self.mapping, :]
    out_vector[self.velocity_mapping, :] = \
        inv_rotation_matrix @ out_vector[self.velocity_mapping, :]

    out_vector[self.mapping, :] = out_vector[self.mapping, :] + \
        self.translation_offset

    return out_vector

  def rvs(self, num_samples=1, **kwargs) -> Union[StateVector, StateVectors]:
    out = super().rvs(num_samples, **kwargs)
    out = np.array([[Elevation(0)], [Bearing(0)], [0.], [0.]]) + out
    return out


class RangeRangeRateBinning(CartesianToElevationBearingRangeRate):
  r"""This is a class implementation of a time-invariant measurement model, \
  where measurements are assumed to be in the form of elevation \
  (:math:`\theta`),  bearing (:math:`\phi`), range (:math:`r`) and
  range-rate (:math:`\dot{r}`), with Gaussian noise in each dimension and the
  range and range-rate are binned based on the
  range resolution and range-rate resolution respectively.

  The model is described by the following equations:

  .. math::

    \vec{y}_t = h(\vec{x}_t, \vec{v}_t)

  where:

  * :math:`\vec{y}_t` is a measurement vector of the form:

  .. math::

    \vec{y}_t = \begin{bmatrix}
              \theta \\
              \phi \\
              r \\
              \dot{r}
          \end{bmatrix}

  * :math:`h` is a non-linear model function of the form:

  .. math::

    h(\vec{x}_t,\vec{v}_t) = \begin{bmatrix}
              \textrm{asin}(\mathcal{z}/\sqrt{\mathcal{x}^2 + \mathcal{y}^2 +\mathcal{z}^2}) \\
              \textrm{atan2}(\mathcal{y},\mathcal{x}) \\
              \sqrt{\mathcal{x}^2 + \mathcal{y}^2 + \mathcal{z}^2} \\
              (x\dot{x} + y\dot{y} + z\dot{z})/\sqrt{x^2 + y^2 + z^2}
              \end{bmatrix} + \vec{v}_t

  * :math:`\vec{v}_t` is Gaussian distributed with covariance :math:`R`, i.e.:

  .. math::

    \vec{v}_t \sim \mathcal{N}(0,R)

  .. math::

    R = \begin{bmatrix}
          \sigma_{\theta}^2 & 0 & 0 & 0\\
          0 & \sigma_{\phi}^2 & 0 & 0\\
          0 & 0 & \sigma_{r}^2 & 0\\
          0 & 0 & 0 & \sigma_{\dot{r}}^2
          \end{bmatrix}

  The covariances for radar are determined by different factors. The angle error
  is affected by the radar beam width. Range error is affected by the SNR and pulse bandwidth.
  The error for the range rate is dependent on the dwell time.
  The range and range rate are binned to the centre of the cell using

  .. math::

      x = \textrm{floor}(x/\Delta x)*\Delta x + \frac{\Delta x}{2}

  The :py:attr:`mapping` property of the model is a 3 element vector, \
  whose first (i.e. :py:attr:`mapping[0]`), second (i.e. \
  :py:attr:`mapping[1]`) and third (i.e. :py:attr:`mapping[2]`) elements \
  contain the state index of the :math:`x`, :math:`y` and :math:`z`  \
  coordinates, respectively.

  The :py:attr:`velocity_mapping` property of the model is a 3 element vector, \
  whose first (i.e. :py:attr:`velocity_mapping[0]`), second (i.e. \
  :py:attr:`velocity_mapping[1]`) and third (i.e. :py:attr:`velocity_mapping[2]`) elements \
  contain the state index of the :math:`\dot{x}`, :math:`\dot{y}` and :math:`\dot{z}`  \
  coordinates, respectively.

  Note
  ----
  This class implementation assumes a 3D cartesian space, it therefore \
  expects a 6D state space.
  """

  range_res: float = Property(doc="Size of the range bins in m")
  range_rate_res: float = Property(doc="Size of the velocity bins in m/s")

  @property
  def ndim_meas(self):
    return 4

  def function(self, state, noise=False, **kwargs):
    r"""Model function :math:`h(\vec{x}_t,\vec{v}_t)`

    Parameters
    ----------
    state: :class:`~.StateVector`
        An input state vector for the target

    noise: :class:`numpy.ndarray` or bool
        An externally generated random process noise sample (the default is
        ``False``, in which case no noise will be added and no binning takes place
        if ``True``, the output of :attr:`~.Model.rvs` is added and the
        range and range rate are binned)

    Returns
    -------
    :class:`numpy.ndarray` of shape (:py:attr:`~ndim_state`, 1)
        The model function evaluated given the provided time interval.

    """

    out = super().function(state, noise, **kwargs)

    if isinstance(noise, bool) or noise is None:
      if noise:
        out[2] = np.floor(out[2] / self.range_res) * \
            self.range_res + self.range_res/2
        out[3] = np.floor(out[3] / self.range_rate_res) * \
            self.range_rate_res + self.range_rate_res/2

    return out

  def _gaussian_integral(self, a, b, mean, cov):
    # this function is the cumulative probability ranging from a to b for a normal distribution
    return (multivariate_normal.cdf(a, mean=mean, cov=cov)
            - multivariate_normal.cdf(b, mean=mean, cov=cov))

  def _binned_pdf(self, measured_value, mean, bin_size, cov):
    # this function finds the probability density of the bin the measured_value is in
    a = np.floor(measured_value / bin_size) * bin_size + bin_size
    b = np.floor(measured_value / bin_size) * bin_size
    return self._gaussian_integral(a, b, mean, cov)/bin_size

  def pdf(self, state1, state2, **kwargs):
    r"""Model pdf/likelihood evaluation function

    Evaluates the pdf/likelihood of ``state1``, given the state
    ``state2`` which is passed to :meth:`function()`.

    For the first 2 dimensions, this can be written as:

    .. math::

        p = p(y_t | x_t) = \mathcal{N}(y_t; x_t, Q)

    where :math:`y_t` = ``state_vector1``, :math:`x_t` = ``state_vector2``,
     :math:`Q` = :attr:`covar` and :math:`\mathcal{N}` is a normal distribution

    The probability for the binned dimensions, the last 2, can be written as:

    .. math::

        p = P(a \leq \mathcal{N} \leq b)

    In this equation a and b are the edges of the bin.

    Parameters
    ----------
    state1 : :class:`~.State`
    state2 : :class:`~.State`

    Returns
    -------
    : :class:`~.Probability`
        The likelihood of ``state1``, given ``state2``
    """

    # state1 is in measurement space
    # state2 is in state_space
    if (((state1.state_vector[2, 0]-self.range_res/2) / self.range_res).is_integer()
            and ((state1.state_vector[3, 0]-self.range_rate_res/2) /
                 self.range_rate_res).is_integer()):
      mean_vector = self.function(state2, noise=False, **kwargs)
      # pdf for the angles
      az_el_pdf = multivariate_normal.pdf(
          state1.state_vector[:2, 0],
          mean=mean_vector[:2, 0],
          cov=self.covar()[:2])

      # pdf for the binned range and velocity
      range_pdf = self._binned_pdf(
          state1.state_vector[2, 0],
          mean_vector[2, 0],
          self.range_res,
          self.covar()[2])
      velocity_pdf = self._binned_pdf(
          state1.state_vector[3, 0],
          mean_vector[3, 0],
          self.range_rate_res,
          self.covar()[3])
      return Probability(range_pdf * velocity_pdf * az_el_pdf)
    else:
      return Probability(0)


class RangeRangeRateBinningAliasing(RangeRangeRateBinning):
  r"""This is a class implementation of a time-invariant measurement model, \
    where measurements are assumed to be in the form of elevation \
    (:math:`\theta`),  bearing (:math:`\phi`), range (:math:`r`) and
    range-rate (:math:`\dot{r}`), with Gaussian noise in each dimension and the
    range and range-rate are binned based on the
    range resolution and range-rate resolution respectively.

    The model is described by the following equations:

    .. math::

      \vec{y}_t = h(\vec{x}_t, \vec{v}_t)

    where:

    * :math:`\vec{y}_t` is a measurement vector of the form:

    .. math::

      \vec{y}_t = \begin{bmatrix}
                \theta \\
                \phi \\
                r \\
                \dot{r}
            \end{bmatrix}

    * :math:`h` is a non-linear model function of the form:

    .. math::

      h(\vec{x}_t,\vec{v}_t) = \begin{bmatrix}
                \textrm{asin}(\mathcal{z}/\sqrt{\mathcal{x}^2 + \mathcal{y}^2 +\mathcal{z}^2}) \\
                \textrm{atan2}(\mathcal{y},\mathcal{x}) \\
                \sqrt{\mathcal{x}^2 + \mathcal{y}^2 + \mathcal{z}^2} \\
                (x\dot{x} + y\dot{y} + z\dot{z})/\sqrt{x^2 + y^2 + z^2}
                \end{bmatrix} + \vec{v}_t

    * :math:`\vec{v}_t` is Gaussian distributed with covariance :math:`R`, i.e.:

    .. math::

      \vec{v}_t \sim \mathcal{N}(0,R)

    .. math::

      R = \begin{bmatrix}
            \sigma_{\theta}^2 & 0 & 0 & 0\\
            0 & \sigma_{\phi}^2 & 0 & 0\\
            0 & 0 & \sigma_{r}^2 & 0\\
            0 & 0 & 0 & \sigma_{\dot{r}}^2
            \end{bmatrix}

    The covariances for radar are determined by different factors. The angle error
    is affected by the radar beam width. Range error is affected by the SNR and pulse bandwidth.
    The error for the range rate is dependent on the dwell time.
    The range and range rate are binned to the centre of the cell using

    .. math::

        x = \textrm{floor}(x/\Delta x)*\Delta x + \frac{\Delta x}{2}

    The :py:attr:`mapping` property of the model is a 3 element vector, \
    whose first (i.e. :py:attr:`mapping[0]`), second (i.e. \
    :py:attr:`mapping[1]`) and third (i.e. :py:attr:`mapping[2]`) elements \
    contain the state index of the :math:`x`, :math:`y` and :math:`z`  \
    coordinates, respectively.

    The :py:attr:`velocity_mapping` property of the model is a 3 element vector, \
    whose first (i.e. :py:attr:`velocity_mapping[0]`), second (i.e. \
    :py:attr:`velocity_mapping[1]`) and third (i.e. :py:attr:`velocity_mapping[2]`) elements \
    contain the state index of the :math:`\dot{x}`, :math:`\dot{y}` and :math:`\dot{z}`  \
    coordinates, respectively.

    Note
    ----
    This class implementation assumes a 3D cartesian space, it therefore \
    expects a 6D state space.
  """
  range_res: float = Property(doc="Size of each range bin in meters")
  range_rate_res: float = Property(doc="Size of each velocity bin in m/s")
  max_unambiguous_range: float = Property(
      doc="Maximum range before aliasing occurs in meters")
  max_unambiguous_range_rate: float = Property(
      doc="Maximum velocity before aliasing occurs in m/s")

  def function(self, state, noise=False, **kwargs):
    r"""Model function :math:`h(\vec{x}_t,\vec{v}_t)`

        Parameters
        ----------
        state: :class:`~.StateVector`
            An input state vector for the target

        noise: :class:`numpy.ndarray` or bool
            An externally generated random process noise sample (the default is
            ``False``, in which case no noise will be added and no binning takes place
            if ``True``, the output of :attr:`~.Model.rvs` is added and the
            range and range rate are binned)

        Returns
        -------
        :class:`numpy.ndarray` of shape (:py:attr:`~ndim_state`, 1)
            The model function evaluated given the provided time interval.

    """
    out = super().function(state, noise, **kwargs)

    if isinstance(noise, bool) or noise is None:
      if noise:
        # Add aliasing to the range/range rate if it exceeds the unambiguous limits
        out[2] = wrap_to_interval(out[2], 0, self.max_unambiguous_range)
        out[3] = wrap_to_interval(
            out[3], -self.max_unambiguous_range_rate, self.max_unambiguous_range_rate)
        # Bin the range and range rate to the center of the cell
        out[2] = np.floor(out[2] / self.range_res) * \
            self.range_res + self.range_res/2
        out[3] = np.floor(out[3] / self.range_rate_res) * \
            self.range_rate_res + self.range_rate_res/2

    return out
