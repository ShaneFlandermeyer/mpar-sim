import datetime
from ordered_set import OrderedSet
from stonesoup.models.transition.base import TransitionModel
import numpy as np
from stonesoup.base import Property
from stonesoup.types.array import StateVector, CovarianceMatrix
from typing import Optional, Collection
from stonesoup.types.state import GaussianState
from stonesoup.types.numeric import Probability
from stonesoup.simulator.simple import SingleTargetGroundTruthSimulator
from stonesoup.types.groundtruth import GroundTruthPath, GroundTruthState
from stonesoup.buffered_generator import BufferedGenerator

class MultiTargetScenario(SingleTargetGroundTruthSimulator):
    """Target simulator that produces multiple targets.

    Targets are created and destroyed randomly, as defined by the birth rate
    and death probability.
    
    This is very similar to the MultiTargetGroundTruthSimulator built in to stonesoup, but it includes a check to avoid repeated simulation of targets when the timestep is zero, as will frequently be the case when simulating a phased array that can steer multiple beams at once.
    
    TODO: Add an RCS field (currently done outside this class).
    """
    transition_model: TransitionModel = Property(
        doc="Transition Model used as propagator for track.")
    initial_state: GaussianState = Property(doc="Initial state to use to generate states")
    birth_rate: float = Property(
        default=1.0, doc="Rate at which tracks are born. Expected number of occurrences (λ) in "
                         "Poisson distribution. Default 1.0.")
    death_probability: Probability = Property(
        default=0.1, doc="Probability of track dying in each time step. Default 0.1.")
    seed: Optional[int] = Property(default=None, doc="Seed for random number generation."
                                                     " Default None")
    preexisting_states: Collection[StateVector] = Property(
        default=list(), doc="State vectors at time 0 for "
                            "groundtruths which should exist at the start of simulation.")
    initial_number_targets: int = Property(
        default=0, doc="Initial number of targets to be "
                       "simulated. These simulated targets will be made in addition to those "
                       "defined by :attr:`preexisting_states`.")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.seed is not None:
            self.random_state = np.random.RandomState(self.seed)
        else:
            self.random_state = np.random.mtrand._rand

    def _new_target(self, time, random_state, state_vector=None):
        vector = state_vector or \
            self.initial_state.state_vector + \
            self.initial_state.covar @ \
            random_state.randn(self.initial_state.ndim, 1)

        gttrack = GroundTruthPath()
        gttrack.append(GroundTruthState(
            state_vector=vector,
            timestamp=time,
            metadata={"index": self.index})
        )
        return gttrack

    @BufferedGenerator.generator_method
    def groundtruth_paths_gen(self, random_state=None):
        time = self.initial_state.timestamp or datetime.datetime.now()
        random_state = random_state if random_state is not None else self.random_state
        number_steps_remaining = self.number_steps

        if self.preexisting_states or self.initial_number_targets:
            # Use preexisting_states to make some groundtruth paths
            preexisting_paths = OrderedSet(
                self._new_target(time, random_state, state) for state in self.preexisting_states)

            # Simulate more groundtruth paths for the number of initial_simulated_states
            initial_simulated_paths = OrderedSet(
                self._new_target(time, random_state) for _ in range(self.initial_number_targets))

            # Union the two sets
            groundtruth_paths = preexisting_paths | initial_simulated_paths

            number_steps_remaining -= 1
            yield time, groundtruth_paths
            time += self.timestep

        else:
            groundtruth_paths = OrderedSet()

        for _ in range(number_steps_remaining):
            # Random drop tracks
            groundtruth_paths.difference_update(
                gttrack
                for gttrack in groundtruth_paths.copy()
                if random_state.rand() <= self.death_probability)

            # Move tracks forward
            if self.timestep > datetime.timedelta(seconds=0):
              for gttrack in groundtruth_paths:
                  self.index = gttrack[-1].metadata.get("index")
                  trans_state_vector = self.transition_model.function(
                      gttrack[-1], noise=True, time_interval=self.timestep)
                  gttrack.append(GroundTruthState(
                      trans_state_vector, timestamp=time,
                      metadata={"index": self.index}))

            # Random create
            for _ in range(random_state.poisson(self.birth_rate)):
                self.index = 0
                gttrack = self._new_target(time, random_state)
                groundtruth_paths.add(gttrack)

            yield time, groundtruth_paths
            time += self.timestep