import random

import loggers
from TNDP import TNDP, TndpNetwork


class GeneticAlgorithm:

    def __init__(self,
                 tndp: TNDP,
                 initial_population_size,
                 initial_population_network_size,
                 n_generations,
                 elitism_percent=0.2) -> None:

        self.tndp = tndp
        self.population_size = initial_population_size
        self.network_size = initial_population_network_size
        self.n_generations = n_generations
        self.elitism_percent = elitism_percent
        self.elite_size = int(
            initial_population_network_size * elitism_percent)

    def _walk_node(self, v):
        return (v, None)

    def _route_node(self, v, route_id):
        return (self, v, route_id)

    def _remove_duplicate_routes(self, routes):
        unique = []
        seen = set()

        for route in routes:
            sig = tuple(route['path'])  # или route_signature(route)
            if sig not in seen:
                seen.add(sig)
                unique.append(route)

        return unique

    def _get_random_route(self):
        return random.choice(self.tndp.line_pool)

    # --- CROSSOVER ---

    def _tournament_selection(self, population, k=3):
        candidates = random.sample(range(len(population)), k)
        best = min(
            candidates, key=lambda i: self.tndp.evaluate_fitness(population[i]))
        return population[best]

    def _get_parents(self, population, k=3):
        p1 = self._tournament_selection(population, k)
        p2 = self._tournament_selection(population, k)

        while p2 is p1:
            p2 = self._tournament_selection(population, k)

        return p1, p2

    def _crossover(self, parent1: TndpNetwork, parent2: TndpNetwork) -> TndpNetwork:
        cut1 = random.randint(0, len(parent1.routes) - 1)
        cut2 = random.randint(0, len(parent2.routes) - 1)

        # print(f'from parent 1: {cut1}, from parent 2: {cut2}')

        child_routes = parent1.routes[:cut1] + parent2.routes[cut2:]
        child_routes = self._remove_duplicate_routes(child_routes)[
            :self.tndp.max_network_size]
        return TndpNetwork(child_routes)

    # --- MUTATIONS ---

    def _mutate_add(self, network: TndpNetwork, p=0.5):
        if random.random() < p and len(network.routes) < self.tndp.max_network_size:
            network.routes.append(self._get_random_route())
            # print('mutate add')

    def _mutate_replace(self, network: TndpNetwork, p=0.5):
        if random.random() < p and len(network.routes) > 0:
            i = random.randrange(len(network.routes))
            new_route = self._get_random_route()
            network.routes[i] = new_route
            # print('mutate replace')

    def _mutate_remove(self, network: TndpNetwork, p=0.1):
        if random.random() < p and len(network.routes) > 1:
            i = random.randrange(len(network.routes))
            network.routes.pop(i)
            # print('mutate remove')

    def _mutate(self, network: TndpNetwork):
        self._mutate_replace(network)
        self._mutate_add(network)
        self._mutate_remove(network)

        network.routes = self._remove_duplicate_routes(network.routes)

    # --- ALGORITHM ---

    def _generate_initial_population(self):
        population = []
        for i in range(self.population_size):
            routes = [random.choice(self.tndp.line_pool)
                      for i in range(self.network_size)]
            population.append(TndpNetwork(routes))
        return population

    def _get_best_solution(self, population):
        best_network = min(
            population, key=lambda i: self.tndp.evaluate_fitness(i))
        return best_network, self.tndp.evaluate_fitness(best_network)

    def _get_elite(self, population):
        fitnesses = [self.tndp.evaluate_fitness(sol) for sol in population]

        sorted_pop = [
            sol for sol, fit in
            sorted(zip(population, fitnesses), key=lambda x: x[1])
        ]
        return sorted_pop[:self.elite_size]

    def generate_solution(self):
        log = loggers.GALoggerTxt('log.txt', self)

        population = self._generate_initial_population()
        total_best_solution, total_best_fitness = self._get_best_solution(
            population)
        log.log_generation('INITIAL GENERATION',
                           population, total_best_fitness)

        for gen in range(self.n_generations):
            gen_best_solution, gen_best_fitness = self._get_best_solution(
                population)

            log.log_generation(
                f'GEN {gen + 1}', population, total_best_fitness)

            if gen_best_fitness < total_best_fitness:
                total_best_fitness = gen_best_fitness
                total_best_solution = gen_best_solution

            # new_population = []
            new_population = self._get_elite(population)

            while len(new_population) < self.population_size:
                parents = self._get_parents(population, k=3)
                child = self._crossover(parents[0], parents[1])
                self._mutate(child)
                new_population.append(child)

            population = new_population

        log.log_result(total_best_fitness)
        log.close_log()

        return total_best_solution, total_best_fitness
