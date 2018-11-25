from framework.graph_search import *
from framework.ways import *
from .map_problem import MapProblem
from .deliveries_problem_input import DeliveriesProblemInput
from .relaxed_deliveries_problem import RelaxedDeliveriesState, RelaxedDeliveriesProblem
from .map_heuristics import AirDistHeuristic
from typing import Set, FrozenSet, Optional, Iterator, Tuple, Union


class StrictDeliveriesState(RelaxedDeliveriesState):
    """
    An instance of this class represents a state of the strict
     deliveries problem.
    This state is basically similar to the state of the relaxed
     problem. Hence, this class inherits from `RelaxedDeliveriesState`.

    TODO:
        If you believe you need to modify the state for the strict
         problem in some sense, please go ahead and do so.
    """


class StrictDeliveriesProblem(RelaxedDeliveriesProblem):
    """
    An instance of this class represents a strict deliveries problem.
    """

    name = 'StrictDeliveries'

    def __init__(self, problem_input: DeliveriesProblemInput, roads: Roads,
                 inner_problem_solver: GraphProblemSolver, use_cache: bool = True):
        super(StrictDeliveriesProblem, self).__init__(problem_input)
        self.my_relaxed = RelaxedDeliveriesProblem(problem_input)
        self.initial_state = StrictDeliveriesState(
            problem_input.start_point, frozenset(), problem_input.gas_tank_init_fuel)
        self.inner_problem_solver = inner_problem_solver
        self.roads = roads
        self.use_cache = use_cache
        self._init_cache()
        self.problem_input = problem_input

    def _init_cache(self):
        self._cache = {}
        self.nr_cache_hits = 0
        self.nr_cache_misses = 0

    def _insert_to_cache(self, key, val):
        if self.use_cache:
            self._cache[key] = val

    def _get_from_cache(self, key):
        if not self.use_cache:
            return None
        if key in self._cache:
            self.nr_cache_hits += 1
        else:
            self.nr_cache_misses += 1
        return self._cache.get(key)

    def _junction_distance(self, source_junction: Junction, destination_junction: Junction)->float:
        cache_index = (source_junction.index, destination_junction.index)

        if self.use_cache:
            cache_result = self._get_from_cache(cache_index)

            if cache_result is not None:
                return cache_result

        result = self.inner_problem_solver.solve_problem(MapProblem(
            self.roads, source_junction.index, destination_junction.index)).final_search_node.cost

        if self.use_cache:
            self._insert_to_cache(cache_index, result)
        return result

    def expand_state_with_costs(self, state_to_expand: GraphProblemState) -> Iterator[Tuple[GraphProblemState, float]]:
        """
        TODO: implement this method!
        This method represents the `Succ: S -> P(S)` function of the strict deliveries problem.
        The `Succ` function is defined by the problem operators as shown in class.
        The relaxed problem operators are defined in the assignment instructions.
        It receives a state and iterates over the successor states.
        Notice that this is an *Iterator*. Hence it should be implemented using the `yield` keyword.
        For each successor, a pair of the successor state and the operator cost is yielded.
        """
        assert isinstance(state_to_expand, StrictDeliveriesState)

        # list of possible legal stop points from the current junction
        legal_junctions = self.possible_stop_points.difference(
            state_to_expand.dropped_so_far, {state_to_expand.current_location})

        # for every legal stop point we have enough fuel to get to
        for junction in [j for j in legal_junctions if self._junction_distance(state_to_expand.current_location, j) <= state_to_expand.fuel]:
            # just here to make state in this scope
            state = None
            # if the junction is a gas station, make the state accordingly
            if junction in self.gas_stations:
                state = StrictDeliveriesState(
                    junction, state_to_expand.dropped_so_far, self.gas_tank_capacity)
            # if the junction is a drop point, make the state accordingly
            else:
                new_fuel = state_to_expand.fuel - \
                    self._junction_distance(
                        state_to_expand.current_location, junction)

                state = StrictDeliveriesState(
                    junction, state_to_expand.dropped_so_far.union({junction}), new_fuel)

            yield state,  self._junction_distance(
                state_to_expand.current_location, junction)

    def is_goal(self, state: GraphProblemState) -> bool:
        """
        This method receives a state and returns whether this state is a goal.
        TODO: implement this method!
        """
        assert isinstance(state, StrictDeliveriesState)

        return super(StrictDeliveriesProblem, self).is_goal(state)
