from TNDP import TNDP, TndpNetwork

import random
import numpy as np

from pymoo.core.problem import Problem
from pymoo.core.repair import Repair
from pymoo.core.sampling import Sampling
from pymoo.core.mutation import Mutation
from pymoo.core.crossover import Crossover


def repair_individual(x, max_routes):
    unique_routes = list(dict.fromkeys([r for r in x if r != -1]))

    if len(unique_routes) > max_routes:
        unique_routes = unique_routes[:max_routes]

    x_new = np.full(len(x), -1, dtype=int)
    x_new[:len(unique_routes)] = unique_routes

    return x_new


class TndpProblem(Problem):
    def __init__(self, K, TNDP: TNDP):
        super().__init__(
            n_var=K,
            n_obj=2,
            xl=-1,
            xu=len(TNDP.line_pool)-1,
            type_var=int
        )
        self.TNDP = TNDP

    def _evaluate(self, x, out, *args, **kwargs):
        fitness = []

        for individual in x:
            individual_network = self._ids_to_network(individual)
            f1 = self.TNDP.evaluate_total_time(individual_network)
            f2 = self.TNDP.evaluate_cost(individual_network)
            fitness.append([f1, f2])

        out["F"] = np.array(fitness)
        
    def _ids_to_network(self, route_ids):
        routes = []
        for id in route_ids:
            routes.append(self.TNDP.line_pool[id])
        return TndpNetwork(routes)


class TndpNetworkRepair(Repair):
    def __init__(self, max_routes) -> None:
        super().__init__()
        self.max_routes = max_routes

    def _do(self, problem, X, **kwargs):
        for i in range(len(X)):
            X[i] = repair_individual(X[i], self.max_routes)
        return X


class TndpSampling(Sampling):
    def __init__(self, K, TNDP: TNDP) -> None:
        super().__init__()
        self.TNDP = TNDP
        self.K = K
        self.vtype = "int"
        self.repair = None

    def _do(self, problem, n_samples, **kwargs):

        X = np.full((n_samples, self.K), -1, dtype=int)

        for i in range(n_samples):
            m = random.randint(1, self.K)
            routes = random.sample(range(len(self.TNDP.line_pool)), m)
            X[i, :m] = routes

        return X


class TndpMutation(Mutation):
    def __init__(
        self,
        all_routes,
        max_routes,
        p_add=0.3,
        p_remove=0.3,
        p_replace=0.4
    ):
        super().__init__()
        self.all_routes = all_routes
        self.max_routes = max_routes
        self.p_add = p_add
        self.p_remove = p_remove
        self.p_replace = p_replace

    def _get_active_routes(self, x):
        return [r for r in x if r != -1]

    def _get_probabilities(self, x):
        size = np.sum(x != -1)
        if size < self.max_routes / 2:
            return [0.5, 0.1, 0.4]
        else:
            return [0.1, 0.3, 0.6]

    def _do(self, problem, X, **kwargs):
        for i in range(len(X)):
            X[i] = self._mutate_individual(X[i])
        return X

    def _mutate_individual(self, x):
        routes = self._get_active_routes(x)
        free_slots = x.tolist().count(-1)

        mutation_type = random.choices(
            ["add", "remove", "replace"],
            weights=self._get_probabilities(x),
            k=1
        )[0]

        if mutation_type == "add":
            return self._add_route(x, routes, free_slots)

        if mutation_type == "remove":
            return self._remove_route(x, routes)

        if mutation_type == "replace":
            return self._replace_route(x, routes)

        return x

    def _add_route(self, x, routes, free_slots):
        if free_slots == 0:
            return x 

        available = list(set(range(len(self.all_routes))) - set(routes))
        if not available:
            return x

        new_route = random.choice(available)

        idx = np.where(x == -1)[0][0]
        x[idx] = new_route
        return x

    def _remove_route(self, x, routes):
        if len(routes) == 0:
            return x

        route_to_remove = random.choice(routes)
        idx = np.where(x == route_to_remove)[0][0]
        x[idx] = -1
        return x

    def _replace_route(self, x, routes):
        if len(routes) == 0:
            return x

        available = list(set(range(len(self.all_routes))) - set(routes))
        if not available:
            return x

        old_route = random.choice(routes)
        new_route = random.choice(available)

        idx = np.where(x == old_route)[0][0]
        x[idx] = new_route
        return x


class TndpCrossover(Crossover):
    def __init__(self):
        super().__init__(2, 2)

    def _get_active(self, x):
        return [r for r in x if r != -1]

    def _do(self, problem, X, **kwargs):
        n_matings, n_var = X.shape[1], X.shape[2] 
        offspring = np.full((self.n_offsprings, n_matings, n_var), -1, dtype=int)

        for i in range(n_matings):
            parent1 = X[0, i, :]
            parent2 = X[1, i, :]

            child1, child2 = self._crossover_pair(parent1, parent2, n_var)

            offspring[0, i, :] = child1
            offspring[1, i, :] = child2

        return offspring

    def _crossover_pair(self, p1, p2, max_routes):
        routes1 = self._get_active(p1)
        routes2 = self._get_active(p2)

        child1_routes = self._mix_routes(routes1, routes2, max_routes)
        child2_routes = self._mix_routes(routes2, routes1, max_routes)

        child1 = self._to_vector(child1_routes, max_routes)
        child2 = self._to_vector(child2_routes, max_routes)

        return child1, child2

    def _mix_routes(self, r1, r2, max_routes):
        k1 = random.randint(1, len(r1))
        k2 = random.randint(1, len(r2))

        part1 = random.sample(r1, k1) if r1 else []
        part2 = random.sample(r2, k2) if r2 else []

        combined = list(set(part1 + part2))

        if len(combined) > max_routes:
            combined = random.sample(combined, max_routes)

        return combined

    def _to_vector(self, routes, n_vars):
        x = np.full(n_vars, -1)
        x[:len(routes)] = routes
        # np.random.shuffle(x)
        return x
    
    
def _get_active_routes(x):
    return [r for r in x if r != -1]

def solution_to_Network(TNDP: TNDP, solution):
    route_ids = _get_active_routes(solution)
    routes = [TNDP.line_pool[id] for id in route_ids]
    return TndpNetwork(routes)
        
    
    
