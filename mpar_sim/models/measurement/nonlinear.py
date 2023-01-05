
import numpy as np
# TODO: Replicate this and remove the dependency on stonesoup
from stonesoup.base import clearable_cached_property

from mpar_sim.common import wrap_to_interval
from mpar_sim.common.coordinate_transform import cart2sph, rotx, roty, rotz, sph2cart
from mpar_sim.models.measurement.base import MeasurementModel


class RangeRangeRateBinningAliasing(MeasurementModel):
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

  def __init__(self,
               translation_offset: np.ndarray = np.zeros((3, 1)),
               rotation_offset: np.ndarray = np.zeros((3, 1)),
               velocity: np.ndarray = np.zeros((3, 1)),
               mapping: tuple[int, int, int] = (0, 2, 4),
               velocity_mapping: tuple[int, int, int] = (1, 3, 5),
               noise_covar: np.ndarray = np.eye(4),
               range_res: float = 1,
               range_rate_res: float = 1,
               max_unambiguous_range: float = np.inf,
               max_unambiguous_range_rate: float = np.inf,
               ndim_state: int = 6,
               ):
    self.translation_offset = translation_offset
    self.mapping = mapping
    self.rotation_offset = rotation_offset
    self.velocity_mapping = velocity_mapping
    self.velocity = velocity
    self.noise_covar = noise_covar
    self.ndim_state = ndim_state
    self.range_res = range_res
    self.range_rate_res = range_rate_res
    self.max_unambiguous_range = max_unambiguous_range
    self.max_unambiguous_range_rate = max_unambiguous_range_rate

    # This is constant
    self.ndim_meas = self.ndim = 4

  def function(self, state: np.ndarray, noise: bool = False,):
    r"""Model function :math:`h(\vec{x}_t,\vec{v}_t)`

        Parameters
        ----------
        state: :class:`~.np.ndarray`
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
    if noise:
      meas_noise = self.rvs(
          num_samples=state.state_vector.shape[1])
    else:
      meas_noise = 0

    # Account for origin offset in position to enable range and angles to be determined
    xyz_pos = state.state_vector[self.mapping, :] - self.translation_offset
    # Rotate coordinates based upon the sensor_velocity
    xyz_rot = self.rotation_matrix @ xyz_pos
    # Convert to Spherical
    rho, az, el = cart2sph(xyz_rot[0, :], xyz_rot[1, :], xyz_rot[2, :])
    # Determine the net velocity component in the engagement
    xyz_vel = state.state_vector[self.velocity_mapping, :] - self.velocity
    # Use polar to calculate range rate
    rr = np.einsum('ij,ij->j', xyz_pos, xyz_vel) / \
        np.linalg.norm(xyz_pos, axis=0)

    out = np.array([el, az, rho, rr]) + meas_noise
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

  def inverse_function(self, detection) -> np.ndarray:
    theta, phi, rho, rho_rate = detection.state_vector

    x, y, z = sph2cart(rho, phi, theta)
    # because only rho_rate is known, only the components in
    # x,y and z of the range rate can be found.
    x_rate = np.cos(phi) * np.cos(theta) * rho_rate
    y_rate = np.cos(phi) * np.sin(theta) * rho_rate
    z_rate = np.sin(phi) * rho_rate

    inv_rotation_matrix = np.linalg.inv(self.rotation_matrix)

    out_vector = np.zeros((self.ndim_state, 1))
    out_vector[self.mapping, 0] = x, y, z
    out_vector[self.velocity_mapping, 0] = x_rate, y_rate, z_rate

    out_vector[self.mapping,
               :] = inv_rotation_matrix @ out_vector[self.mapping, :]
    out_vector[self.velocity_mapping, :] = \
        inv_rotation_matrix @ out_vector[self.velocity_mapping, :]

    out_vector[self.mapping, :] = out_vector[self.mapping, :] + \
        self.translation_offset

    return out_vector

  def rvs(self,
          num_samples: int = 1) -> np.ndarray:
    covar = self.noise_covar
    noise = np.random.multivariate_normal(
        np.zeros(self.ndim), covar, num_samples)
    noise = np.atleast_2d(noise).T
    return noise

  @clearable_cached_property('rotation_offset')
  def rotation_matrix(self) -> np.ndarray:
    """3D axis rotation matrix"""
    theta_x = -self.rotation_offset[0, 0]  # roll
    theta_y = self.rotation_offset[1, 0]  # pitch#elevation
    theta_z = -self.rotation_offset[2, 0]  # yaw#azimuth
    return rotz(theta_z) @ roty(theta_y) @ rotx(theta_x)
