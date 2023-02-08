
from typing import List, Optional
import numpy as np
# TODO: Replicate this and remove the dependency on stonesoup
from stonesoup.base import clearable_cached_property

from mpar_sim.common import wrap_to_interval
from mpar_sim.common.coordinate_transform import cart2sph, rotx, roty, rotz, sph2cart
from mpar_sim.common.matrix import jacobian
from mpar_sim.models.measurement.base import MeasurementModel


class NonlinearMeasurementModel(MeasurementModel):
  """Base class for nonlinear measurement models"""


class CartesianToRangeAzElRangeRate(NonlinearMeasurementModel):
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

    The :py:attr:`position_mapping` property of the model is a 3 element vector, \
    whose first (i.e. :py:attr:`position_mapping[0]`), second (i.e. \
    :py:attr:`position_mapping[1]`) and third (i.e. :py:attr:`position_mapping[2]`) elements \
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
               # Sensor kinematic information
               translation_offset: np.ndarray = np.zeros((3,)),
               rotation_offset: np.ndarray = np.zeros((3,)),
               velocity: np.ndarray = np.zeros((3,)),
               # Measurement information
               noise_covar: np.ndarray = np.eye(4),
               range_res: float = 1,
               range_rate_res: float = 1,
               discretize_measurements: bool = True,
               # Ambiguity limits
               max_unambiguous_range: float = np.inf,
               max_unambiguous_range_rate: float = np.inf,
               alias_measurements: bool = True,
               # State mappings
               position_mapping: List[int] = [0, 2, 4],
               velocity_mapping: List[int] = [1, 3, 5],
               ):
    # Sensor kinematic information
    self.translation_offset = translation_offset
    self.position_mapping = position_mapping
    self.rotation_offset = rotation_offset
    self.velocity = velocity

    # Measurement parameters
    self.noise_covar = noise_covar
    self.range_res = range_res
    self.range_rate_res = range_rate_res
    self.discretize_measurements = discretize_measurements

    # Ambiguity limits
    self.max_unambiguous_range = max_unambiguous_range
    self.max_unambiguous_range_rate = max_unambiguous_range_rate
    self.alias_measurements = alias_measurements

    # State mappings
    self.position_mapping = position_mapping
    self.velocity_mapping = velocity_mapping
    # This is constant
    self.ndim_state = 6
    self.ndim_meas = self.ndim = 4

  def function(self, state: np.ndarray, noise: bool = False):
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
    state = state.reshape((self.ndim_state, -1))
    if state.ndim == 1:
      state = state.reshape((-1, 1))

    if noise:
      n_states = state.shape[-1]
      meas_noise = np.zeros((self.ndim_meas, n_states))
      for i in range(n_states):
        if self.noise_covar.ndim > 2:
          covar = self.noise_covar[..., i]
        else:
          covar = self.noise_covar
        meas_noise[:, i] = np.random.multivariate_normal(
            np.zeros(self.ndim), covar)
    else:
      meas_noise = 0

    # Account for origin offset in position to enable range and angles to be determined
    xyz_pos = state[self.position_mapping, :] - \
        self.translation_offset.reshape((-1, 1))
    # Rotate coordinates based upon the sensor_velocity
    xyz_rot = self.rotation_matrix @ xyz_pos
    # Convert to Spherical
    az, el, rho = cart2sph(
        xyz_rot[0, :], xyz_rot[1, :], xyz_rot[2, :], degrees=True)
    # Determine the net velocity component in the engagement
    xyz_vel = state[self.velocity_mapping, :] - self.velocity.reshape((-1, 1))
    # Use polar to calculate range rate
    rr = np.einsum('ij, ij->j', xyz_pos, xyz_vel) / \
        np.linalg.norm(xyz_pos, axis=0)

    out = np.array([az, el, rho, rr]) + meas_noise
    if self.alias_measurements:
      # Add aliasing to the range/range rate if it exceeds the unambiguous limits
      out[2] = wrap_to_interval(out[2], 0, self.max_unambiguous_range)
      out[3] = wrap_to_interval(
          out[3], -self.max_unambiguous_range_rate, self.max_unambiguous_range_rate)
    if self.discretize_measurements:
      # Bin the range and range rate to the center of the cell
      out[2] = np.floor(out[2] / self.range_res) * \
          self.range_res + self.range_res/2
      out[3] = np.floor(out[3] / self.range_rate_res) * \
          self.range_rate_res + self.range_rate_res/2
    return out

  def inverse_function(self, detection) -> np.ndarray:
    azimuth, elevation, range, range_rate = detection.state_vector

    x, y, z = sph2cart(azimuth, elevation, range, degrees=True)
    # because only rho_rate is known, only the components in
    # x,y and z of the range rate can be found.
    x_rate = np.cos(azimuth) * np.cos(elevation) * range_rate
    y_rate = np.cos(azimuth) * np.sin(elevation) * range_rate
    z_rate = np.sin(azimuth) * range_rate

    inv_rotation_matrix = np.linalg.inv(self.rotation_matrix)

    out_vector = np.zeros((self.ndim_state, 1))
    out_vector[self.position_mapping, 0] = x, y, z
    out_vector[self.velocity_mapping, 0] = x_rate, y_rate, z_rate

    out_vector[self.position_mapping,
               :] = inv_rotation_matrix @ out_vector[self.position_mapping, :]
    out_vector[self.velocity_mapping, :] = \
        inv_rotation_matrix @ out_vector[self.velocity_mapping, :]

    out_vector[self.position_mapping, :] = out_vector[self.position_mapping, :] + \
        self.translation_offset

    return out_vector

  def covar(self):
    return self.noise_covar

  def jacobian(self, state: Optional[np.ndarray]) -> np.ndarray:
    jacobian = np.zeros((self.ndim_meas, self.ndim_state))

    # Compute position portion of jacobian
    # TODO: Check that r_xy and r_xyz > 0.
    pos = state[self.position_mapping, ...].ravel()
    relative_pos = pos - self.translation_offset.ravel()
    relative_pos = np.dot(self.rotation_matrix, relative_pos)
    x, y, z = relative_pos
    r_xy = np.sqrt(x**2 + y**2)
    r_xyz = np.sqrt(x**2 + y**2 + z**2)
    A = np.zeros((3, 3))
    A[0, :] = np.dot(self.rotation_matrix, np.array([-y, x, 0]) / r_xy**2)
    A[1, :] = np.dot(self.rotation_matrix, np.array(
        [-x*z, -y*z, r_xy**2]) / (r_xy * r_xyz**2))
    A[2, :] = np.dot(self.rotation_matrix, np.array([x, y, z]) / r_xyz)
    # Convert to degrees and store the result
    A[:2, :] *= 180 / np.pi
    jacobian[:-1, ::2] = A

    # Compute range rate portion of jacobian
    # TODO: Check that r > 0
    vel = state[self.velocity_mapping, ...].ravel()
    relative_vel = vel - self.velocity.ravel()
    rdot_to_x = (relative_vel[0] * r_xyz -
                 np.dot(relative_pos, relative_vel) * x / r_xyz) / r_xyz**2
    rdot_to_xdot = x / r_xyz
    rdot_to_y = (relative_vel[1] * r_xyz - np.dot(relative_pos, relative_vel) *
                 y / r_xyz) / r_xyz**2
    rdot_to_ydot = y / r_xyz
    rdot_to_z = (relative_vel[2] * r_xyz - np.dot(relative_pos, relative_vel) *
                 z / r_xyz) / r_xyz**2
    rdot_to_zdot = z / r_xyz
    jacobian[-1, :] = [rdot_to_x, rdot_to_xdot,
                       rdot_to_y, rdot_to_ydot, rdot_to_z, rdot_to_zdot]

    return jacobian

  @clearable_cached_property('rotation_offset')
  def rotation_matrix(self) -> np.ndarray:
    """3D axis rotation matrix"""
    theta_x = -self.rotation_offset.ravel()[0]  # roll
    theta_y = self.rotation_offset.ravel()[1]  # pitch#elevation
    theta_z = -self.rotation_offset.ravel()[2]  # yaw#azimuth
    return rotz(theta_z) @ roty(theta_y) @ rotx(theta_x)
