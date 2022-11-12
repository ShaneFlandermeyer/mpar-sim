import copy
from mpar_sim.beam.common import beam_broadening_factor
from mpar_sim.beam.beam import RectangularBeam
from mpar_sim.common.albersheim import albersheim_pd
from mpar_sim.common.coordinate_transform import cart2sph, sph2cart, rotx, roty, rotz
from stonesoup.sensor.sensor import Sensor
from typing import Set
from stonesoup.types.groundtruth import GroundTruthState
from stonesoup.types.detection import TrueDetection
from stonesoup.models.measurement import MeasurementModel
from stonesoup.types.state import StateVector
from stonesoup.base import Property
from scipy import constants
from stonesoup.models.measurement.nonlinear import RangeRangeRateBinning
from stonesoup.sensor.radar.radar import RadarElevationBearingRangeRate
from mpar_sim.look import Look
from typing import Callable, List, Optional, Tuple, Union
from scipy import constants
import numpy as np


class MultiBeamRadar(Sensor):
  """An active electronically scanned array (AESA) radar sensor"""

  # Motion and orientation parameters
  rotation_offset: StateVector = Property(
      default=StateVector([0, 0, 0]),
      doc="A 3x1 array of angles (rad), specifying the radar orientation in terms of the "
      "counter-clockwise rotation around the :math:`x,y,z` axis. i.e Roll, Pitch and Yaw. "
      "Default is ``StateVector([0, 0, 0])``")
  position_mapping: Tuple[int, int, int] = Property(
      default=(0, 1, 2),
      doc="Mapping between or positions and state "
          "dimensions. [x,y,z]")
  measurement_model: MeasurementModel = Property(
      default=RadarElevationBearingRangeRate(
          position_mapping=(0, 2, 4),
          velocity_mapping=(1, 4, 5),
          noise_covar=np.array([0, 0, 0, 0])),
      doc="The measurement model used to generate "
      "measurements. By default, this object measures range, range rate, azimuth, and elevation with no noise."
  )
  # Array parameters
  n_elements_x: int = Property(
      default=16,
      doc="The number of horizontal array elements")
  n_elements_y: int = Property(
      default=16,
      doc="The number of vertical array elements")
  element_spacing: float = Property(
      default=0.5,
      doc="The spacing between array elements (m)")
  element_tx_power: float = Property(
      default=10,
      doc="Tx power of each element (W)")
  # System parameters
  center_frequency: float = Property(
      default=3e9,
      doc="Transmit frequency of the array")
  system_temperature: float = Property(
      default=290,
      doc="System noise temperature (K)")
  noise_figure: float = Property(
      default=4,
      doc="Receiver noise figure (dB)")
  # Scan settings
  beam_shape: Callable = Property(
      default=RectangularBeam,
      doc="Object describing the shape of the beam.")
  # Detections settings
  false_alarm_rate: float = Property(
      default=1e-6,
      doc="Probability of false alarm")

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.wavelength = constants.c / self.center_frequency
    # TODO: Handle subarray resource allocation in a separate object.

  @ measurement_model.getter
  def measurement_model(self):
    measurement_model = copy.deepcopy(self._property_measurement_model)
    measurement_model.translation_offset = self.position.copy()
    measurement_model.rotation_offset = self.rotation_offset.copy()
    return measurement_model

  @property
  def _rotation_matrix(self):
    """
    3D rotation matrix for converting target pointing vectors to the sensor frame

    Returns
    -------
    np.ndarray
      (3,3) 3D rotation matrix
    """
    theta_x = -self.rotation_offset[0, 0]  # Roll
    theta_y = -self.rotation_offset[1, 0]  # Pitch/elevation
    theta_z = -self.rotation_offset[2, 0]  # Yaw/azimuth

    return rotz(theta_z) @ roty(theta_y) @ rotx(theta_x)

  def load_look(self, look: Look):
    """
    Allocate resources for the given radar job

    Parameters
    ----------
    look: Look
      The radar job to be scheduled and executed. The following parameters must be present in the job object:
        - bandwidth
        - pulsewidth
        - prf
        - n_pulses
        - azimuth_beamwidth
        - elevation_beamwidth
        - azimuth_steering_angle
        - elevation_steering_angle
        - tx_power
    """
    self.n_pulses = look.n_pulses
    # Compute range/velocity resolutions
    self.range_resolution = constants.c / (2 * look.bandwidth)
    self.velocity_resolution = (
        self.wavelength / 2) * (look.prf / look.n_pulses)

    # Compute ambiguity limits
    self.max_unambiguous_range = constants.c / (2 * look.prf)
    self.max_unambiguous_radial_speed = (self.wavelength / 2) * (look.prf / 2)

    # Create a new beam from the parameter set
    # NOTE: The beam object assumes you have already calculated any beam broadening terms before passing the beamwidths to the objects.
    az_broadening, el_broadening = beam_broadening_factor(
        look.azimuth_steering_angle,
        look.elevation_steering_angle)
    effective_az_beamwidth = look.azimuth_beamwidth * az_broadening
    effective_el_beamwidth = look.elevation_beamwidth * el_broadening
    self.beam = self.beam_type(
        azimuth_beamwidth=effective_az_beamwidth,
        elevation_beamwidth=effective_el_beamwidth,
        azimuth_steering_angle=look.azimuth_steering_angle,
        elevation_steering_angle=look.elevation_steering_angle,
    )

    # Compute the loop gain (the part of the radar range equation that doesn't depend on the target)
    pulse_compression_gain = look.bandwidth * look.pulsewidth
    n_elements_total = int(look.tx_power / self.element_tx_power)
    noise_power = constants.Boltzmann * self.system_temperature * \
        self.noise_figure * look.bandwidth * n_elements_total
    self.loop_gain = look.n_pulses * pulse_compression_gain * look.tx_power * \
        self.beam.gain**2 * self.wavelength**2 / ((4*np.pi)**3 * noise_power)

    # TODO: For now, assume a measurement model with no measurement noise
    self.measurement_model = RangeRangeRateBinning(
        range_res=self.range_resolution,
        range_rate_res=self.velocity_resolution,
        ndim_state=6,
        mapping=[0, 2, 4],
        velocity_mapping=[1, 3, 5],
        noise_covar=np.array([0, 0, 0, 0])
    )

  def measure(self, ground_truths: Set[GroundTruthState], noise: Union[np.ndarray, bool] = True, **kwargs) -> set[TrueDetection]:
    # TODO: Should be able to mostly merge logic between radar detection generator and AESARadar object
    # TODO: Add false alarms to this model
    detections = set()

    measurement_model = copy.deepcopy(self.measurement_model)

    # Loop through the targets and generate detections
    for truth in ground_truths:
      # Get the position of the target in the radar coordinate frame
      relative_pos = truth.state_vector[self.position_mapping,
                                        :] - self.position
      relative_pos = self._rotation_matrix @ relative_pos

      # Convert target position to spherical coordinates
      [r, target_az, target_el] = cart2sph(*relative_pos)

      # Compute target's az/el relative to the beam center
      relative_az = target_az - self.beam.azimuth_steering_angle
      relative_el = target_el - self.beam.elevation_steering_angle
      # Compute loss due to the target being off-centered in the beam
      beam_shape_loss_db = self.beam.shape_loss(relative_az, relative_el)

      snr = 10*np.log10(self.loop_gain) + 10*np.log10(truth.rcs) - \
          40*np.log10(r) - beam_shape_loss_db
          
      # Probability of detection
      pfa = self.false_alarm_rate
      N = self.n_pulses
      pd = albersheim_pd(snr, pfa, N)
      
      # TODO: Use rand() to determine if there were any detections
      # TODO: Account for false alarms


if __name__ == '__main__':
  r = MultiBeamRadar(
      position=StateVector([0, 0, 0])
  )
#   r.load_look(Look())
  print(r.measure(set()))
#   print(r.measurement_model)
