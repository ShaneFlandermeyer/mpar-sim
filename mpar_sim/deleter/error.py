from typing import Sequence, Tuple

import numpy as np

from stonesoup.base import Property
from stonesoup.deleter.base import Deleter

from mpar_sim.common.coordinate_transform import cart2sph_covar


class AngularErrorDeleter(Deleter):
  """ 
  Track deleter based on angular estimation error.

  Deletes tracks whose angular estimation error exceeds some threshold.
  """

  error_thresh: float = Property(doc="Error threshold")
  position_mapping: Tuple[int, int, int] = Property(
      default=(0, 2, 4),
      doc="Mapping between or positions and state "
          "dimensions. [x,y,z]")
  mapping: Sequence[int] = Property(default=None,
                                    doc="Track state vector indices whose corresponding "
                                    "error is to be considered. Defaults to"
                                    "None, whereby the entire track covariance is "
                                    "considered.")

  def check_for_deletion(self, track, **kwargs):
    """Check if a given track should be deleted

    A track is flagged for deletion if the error of the angular part of its state covariance matrix is higher than :py:attr:`~error_thresh`.

    Parameters
    ----------
    track : Track
        A track object to be checked for deletion.

    Returns
    -------
    bool
        `True` if track should be deleted, `False` otherwise.
    """
    # Convert the
    position_xyz = track.state.state_vector[self.position_mapping, :]
    position_covar = track.state.covar[self.position_mapping,
                                       :][:, self.position_mapping]
    sph_covar = cart2sph_covar(position_covar, *position_xyz)
    diagonals = np.diag(sph_covar)

    if self.mapping:
      el_error = diagonals[self.mapping[0]]
      az_error = diagonals[self.mapping[1]]
    else:
      el_error = diagonals[0]
      az_error = diagonals[1]

    if az_error > self.error_thresh or el_error > self.error_thresh:
      return True
    return False
