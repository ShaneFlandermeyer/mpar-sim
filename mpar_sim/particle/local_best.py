# -*- coding: utf-8 -*-

r"""
A local-best Particle Swarm Optimization (gbest PSO) algorithm.

It takes a set of candidate solutions, and tries to find the best
solution using a position-velocity update method. Uses a
ring-topology where each particle is attracted to the best
performing particle in its neighborhood.

Unlike the pyswarms version, this class updates the swarm incrementally and does not specify an upper limit on the number of iterations. This makes it possible to use in a "streaming" context, e.g., for the duration of a phased array's operation.

The position update can be defined as:

.. math::

   x_{i}(t+1) = x_{i}(t) + v_{i}(t+1)

Where the position at the current timestep :math:`t` is updated using
the computed velocity at :math:`t+1`. Furthermore, the velocity update
is defined as:

.. math::

   v_{ij}(t + 1) = w * v_{ij}(t) + c_{1}r_{1j}(t)[y_{ij}(t) − x_{ij}(t)]
                   + c_{2}r_{2j}(t)[\hat{y}_{j}(t) − x_{ij}(t)]

Here, :math:`c1` and :math:`c2` are the cognitive and social parameters
respectively. They control the particle's behavior given two choices: (1) to
follow its *personal best* or (2) follow the swarm's *global best* position.
Overall, this dictates if the swarm is explorative or exploitative in nature.
In addition, a parameter :math:`w` controls the inertia of the swarm's
movement.

An example usage is as follows:

.. code-block:: python

    import pyswarms as ps
    from pyswarms.utils.functions import single_obj as fx

    # Set-up hyperparameters
    options = {'c1': 0.5, 'c2': 0.3, 'w':0.9}

    # Call instance of GlobalBestPSO
    optimizer = ps.single.GlobalBestPSO(n_particles=10, dimensions=2,
                                        options=options)

    # Perform optimization
    for i in range(100):
      stats = optimizer.optimize(fx.sphere)

This algorithm was adapted from the earlier works of J. Kennedy and
R.C. Eberhart in Particle Swarm Optimization [IJCNN1995]_.

.. [IJCNN1995] J. Kennedy and R.C. Eberhart, "Particle Swarm Optimization,"
    Proceedings of the IEEE International Joint Conference on Neural
    Networks, 1995, pp. 1942-1948.
"""

import logging
import multiprocessing as mp
import time
from typing import Tuple

import numpy as np
from pyswarms.backend.handlers import (BoundaryHandler, OptionsHandler,
                                       VelocityHandler)
from pyswarms.backend.operators import (compute_objective_function,
                                        compute_pbest)
from pyswarms.backend.topology import Star, Ring
from pyswarms.base.base_single import SwarmOptimizer
from pyswarms.utils.reporter import Reporter


class IncrementalLocalBestPSO(SwarmOptimizer):
  """Initialize the swarm

    Attributes
    ----------
    n_particles : int
        number of particles in the swarm.
    dimensions : int
        number of dimensions in the space.
    options : dict with keys :code:`{'c1', 'c2', 'w'}`
        a dictionary containing the parameters for the specific
        optimization technique.
            * c1 : float
                cognitive parameter
            * c2 : float
                social parameter
            * w : float
                inertia parameter
    bounds : tuple of numpy.ndarray, optional
        a tuple of size 2 where the first entry is the minimum bound while
        the second entry is the maximum bound. Each array must be of shape
        :code:`(dimensions,)`.
    oh_strategy : dict, optional, default=None(constant options)
        a dict of update strategies for each option.
    bh_strategy : str
        a strategy for the handling of out-of-bounds particles.
    velocity_clamp : tuple, optional
        a tuple of size 2 where the first entry is the minimum velocity and
        the second entry is the maximum velocity. It sets the limits for
        velocity clamping.
    vh_strategy : str
        a strategy for the handling of the velocity of out-of-bounds particles.
    center : list (default is 1.0)
        an array of size :code:`dimensions`
    init_pos : numpy.ndarray, optional
        option to explicitly set the particles' initial positions. Set to
        :code:`None` if you wish to generate the particles randomly.
    """

  def __init__(
      self,
      n_particles,
      dimensions,
      options,
      bounds=None,
      oh_strategy=None,
      bh_strategy="reflective",
      velocity_clamp=None,
      vh_strategy="unmodified",
      center=1.0,
      init_pos=None,
  ) -> None:
    super().__init__(
        n_particles=n_particles,
        dimensions=dimensions,
        options=options,
        bounds=bounds,
        velocity_clamp=velocity_clamp,
        center=center,
        init_pos=init_pos,
    )

    if oh_strategy is None:
      oh_strategy = {}
    # Initialize logger
    self.rep = Reporter(logger=logging.getLogger(__name__))
    # Initialize the resettable attributes
    self.reset()

    # Initialize the topology
    self.top = Ring(static=True)
    self.bh = BoundaryHandler(strategy=bh_strategy)
    self.vh = VelocityHandler(strategy=vh_strategy)
    self.oh = OptionsHandler(strategy=oh_strategy)
    self.name = __name__

    # Reset memory-based items
    self.swarm.pbest_cost = np.full(self.swarm_size[0], np.inf)
    self.iter_count = 0
    # Populate memory of the handlers
    self.bh.memory = self.swarm.position
    self.vh.memory = self.swarm.velocity

  def optimize(self, objective_func, iters=None, n_processes=None, **kwargs) -> Tuple[np.ndarray]:
    # Set up pool of processes for parallel evaluation
    pool = mp.Pool(n_processes) if n_processes else None

    # Compute cost for current position
    self.swarm.current_cost = compute_objective_function(self.swarm,
                                                         objective_func,
                                                         pool=pool,
                                                         **kwargs)

    # Compute global best position/cost
    self.swarm.best_pos, self.swarm.best_cost = self.top.compute_gbest(
        self.swarm, k=20, p=2)

    # Save to history
    hist = self.ToHistory(
        best_cost=self.swarm.best_cost,
        mean_pbest_cost=np.mean(self.swarm.pbest_cost),
        mean_neighbor_cost=self.swarm.best_cost,
        position=self.swarm.position,
        velocity=self.swarm.velocity,
    )
    self._populate_history(hist)

    # Perform options update
    if iters is not None:
      self.swarm.options = self.oh(
          self.options, iternow=self.iter_count, itermax=iters
      )

    # Perform velocity and position updates
    self.swarm.velocity = self.top.compute_velocity(
        self.swarm, self.velocity_clamp, self.vh, self.bounds
    )
    self.swarm.position = self.top.compute_position(
        self.swarm, self.bounds, self.bh
    )

    if pool:
      pool.close()

    self.iter_count += 1
    return (self.swarm.best_cost, self.swarm.best_pos)

  def reset(self):
      super().reset()
      self.swarm.pbest_cost = np.full(self.swarm_size[0], np.inf)