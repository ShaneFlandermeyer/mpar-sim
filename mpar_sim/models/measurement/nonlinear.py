
from typing import List, Optional, Union


from mpar_sim.common import wrap_to_interval
from mpar_sim.common.coordinate_transform import cart2sph, rotx, roty, rotz, sph2cart
from mpar_sim.models.measurement import NonlinearMeasurementModel
from scipy.spatial.transform import Rotation
import numpy as np


class CartesianToRangeAzElVelocity(NonlinearMeasurementModel):
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
               rotation_sequence: str = 'zyx',
               # Measurement information
               noise_covar: np.ndarray = np.eye(4),
               range_resolution: float = None,
               velocity_resolution: float = None,
               # Ambiguity limits
               unambiguous_range: float = None,
               unambiguous_velocity: float = None,
               # State mappings
               position_mapping: List[int] = [0, 2, 4],
               velocity_mapping: List[int] = [1, 3, 5],
               # Optional settings
               discretize_measurements: bool = False,
               alias_measurements: bool = False,
               seed: int = np.random.randint(0, 2**32-1)
               ):
    # Sensor kinematic information
    self.translation_offset = translation_offset
    self.position_mapping = position_mapping
    self.rotation_offset = rotation_offset
    self.rotation_sequence = rotation_sequence
    self.velocity = velocity
    # Measurement parameters
    self.noise_covar = noise_covar
    self.range_res = range_resolution
    self.velocity_res = velocity_resolution
    # Ambiguity limits
    self.unambiguous_range = unambiguous_range
    self.unambiguous_velocity = unambiguous_velocity
    # State mappings
    self.position_mapping = position_mapping
    self.velocity_mapping = velocity_mapping
    # Optional settings
    self.discretize_measurements = discretize_measurements
    self.alias_measurements = alias_measurements
    # Constants
    self.ndim_state = 6
    self.ndim_meas = self.ndim = 4

    self.np_random = np.random.RandomState(seed)

  @property
  def rotation(self):
    return Rotation.from_euler(self.rotation_sequence,
                               np.array(self.rotation_offset).ravel(),
                               degrees=True)

  def __call__(self,
               state: Union[np.ndarray, List[np.ndarray]],
               noise: bool = False):
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
    n_inputs = len(state) if isinstance(state, list) else 1
    state = np.array(state).reshape((n_inputs, self.ndim_state))

    if noise:
      # Standard case: One covariance matrix for all measurements
      if isinstance(self.noise_covar, np.ndarray):
        measurement_noise[:, i] = self.np_random.multivariate_normal(
            mean=np.zeros(self.ndim), cov=self.noise_covar, size=n_inputs).T
      # Special case: One covariance matrix per measurement
      elif isinstance(self.noise_covar, list):
        assert len(self.noise_covar) == n_inputs, \
            "If multiple covariance matrices are provided, the number of matrices must equal the number of input vectors"
        measurement_noise = np.zeros((n_inputs, self.ndim))
        for i in range(n_inputs):
          measurement_noise[i] = self.np_random.multivariate_normal(
              mean=np.zeros(self.ndim), cov=self.noise_covar[i])
    else:
      measurement_noise = 0

    # Transform the state into the sensor frame
    rel_pos = state[:, self.position_mapping] - self.translation_offset
    rel_pos = self.rotation.apply(rel_pos)
    # Convert to Spherical
    az, el, r = cart2sph(*rel_pos.T, degrees=True)
    # Compute radial velocity
    rel_vel = state[:, self.velocity_mapping] - self.velocity
    velocity = np.einsum('ni, ni->n', rel_pos, rel_vel) / \
        np.linalg.norm(rel_pos, axis=1)

    out = np.array([az, el, r, velocity]).T + measurement_noise
    if self.alias_measurements:
      # Add aliasing to the range/range rate if it exceeds the unambiguous limits
      out[:, 2] = wrap_to_interval(out[:, 2], 0, self.unambiguous_range)
      out[:, 3] = wrap_to_interval(
          out[:, 3], -self.unambiguous_velocity, self.unambiguous_velocity)
    if self.discretize_measurements:
      # Bin the range and range rate to the center of the cell
      out[:, 2] = np.floor(out[:, 2] / self.range_res) * \
          self.range_res + self.range_res/2
      out[:, 3] = np.floor(out[:, 3] / self.velocity_res) * \
          self.velocity_res + self.velocity_res/2
    return list(out) if n_inputs > 1 else out.ravel()

  def inverse_function(self, measurement: np.array) -> np.array:
    # Compute the cartesian position
    azimuth, elevation, range, range_rate = measurement.reshape(-1, 1)
    x, y, z = sph2cart(azimuth, elevation, range, degrees=True)

    # Back out the velocity from the range rate
    radar_to_meas = np.array([x, y, z]).reshape(-1, 1)
    radar_to_meas_norm = radar_to_meas / np.linalg.norm(radar_to_meas)
    velocity = range_rate * radar_to_meas_norm + self.velocity.reshape(-1, 1)

    # Form the state vector from the measurement function
    out_vector = np.zeros((self.ndim_state, 1))
    pos_inds = np.array(self.position_mapping)
    vel_inds = np.array(self.velocity_mapping)
    out_vector[pos_inds] = x, y, z
    out_vector[vel_inds] = velocity

    # Rotate the result from the radar frame back into the "global" frame
    rotation = self.rotation
    out_vector[pos_inds] = rotation.apply(
        out_vector[pos_inds].T, inverse=True).T
    out_vector[vel_inds] = rotation.apply(
        out_vector[vel_inds].T, inverse=True).T
    out_vector[pos_inds] += self.translation_offset.reshape((-1, 1))

    return out_vector

  def covar(self):
    return self.noise_covar

  def jacobian(self, state: Optional[np.ndarray]) -> np.ndarray:
    jacobian = np.zeros((self.ndim_meas, self.ndim_state))

    # Compute position portion of jacobian
    # TODO: Check that r_xy and r_xyz > 0.
    rotation = self.rotation
    pos = state[self.position_mapping].ravel()
    relative_pos = pos - self.translation_offset.ravel()
    relative_pos = rotation.apply(relative_pos)
    x, y, z = relative_pos
    r_xy = np.sqrt(x**2 + y**2)
    r_xyz = np.sqrt(x**2 + y**2 + z**2)
    A = np.zeros((3, 3))
    A[0, :] = rotation.apply(np.array([-y, x, 0]) / r_xy**2)
    A[1, :] = rotation.apply(
        np.array([-x*z, -y*z, r_xy**2]) / (r_xy * r_xyz**2))
    A[2, :] = rotation.apply(np.array([x, y, z]) / r_xyz)
    # Convert to degrees and store the result
    A[:2, :] = np.rad2deg(A[:2, :])
    jacobian[:-1, ::2] = A

    # Compute range rate portion of jacobian
    # TODO: Check that r > 0
    vel = state[self.velocity_mapping].ravel()
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
    jacobian[-1, :] = np.array([rdot_to_x, rdot_to_xdot,
                                rdot_to_y, rdot_to_ydot, rdot_to_z, rdot_to_zdot])

    return jacobian
