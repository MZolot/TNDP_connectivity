import random
import networkx as nx

import loggers

COST_OPERATIONAL = 10000
MAX_ROUTES = 20

class Network:
    def __init__(self, routes):
        self.routes = routes
        self.fitness = None

class GeneticAlgorithm:
    def __init__(self,
                 graph,
                 pedestrian_graph,
                 od_matrix,
                 line_pool,
                 population_size,
                 network_size,
                 n_generations,
                 time_coef=10.0,
                 cost_coef=1.0,
                 elitism_percent=0.2) -> None:

        self.graph = graph
        self.pedestrian_graph = pedestrian_graph
        self.od_matrix = od_matrix
        self.line_pool = line_pool
        self.population_size = population_size
        self.network_size = network_size
        self.n_generations = n_generations
        self.time_coef = time_coef
        self.cost_coef = cost_coef
        self.elitism_percent = elitism_percent
        self.elite_size = int(network_size * elitism_percent)

    def walk_node(self, v):
        return (v, None)

    def route_node(self, v, route_id):
        return (self, v, route_id)

    def build_multimodal_graph(self, network: Network):
        network_graph = nx.Graph()

        for u, v, data in self.pedestrian_graph.edges(data=True):
            w = data["weight"]

            network_graph.add_edge(
                self.walk_node(u),
                self.walk_node(v),
                weight=w,
                mode="walk"
            )
        
        transfer_time = 0
        route_id = 0
        for route in network.routes:
            stops = route['path']
            for i in range(len(stops) - 1):
                u = stops[i]
                v = stops[i + 1]
                weight = self.graph.get_edge_data(u ,v)["weight"]

                network_graph.add_edge(self.route_node(u, route_id),
                    self.route_node(v, route_id),
                    weight=weight,
                    mode="ride"
                )

                network_graph.add_edge(
                    self.walk_node(u),
                    self.route_node(u, route_id),
                    weight=transfer_time,
                    mode="transfer"
                )

                network_graph.add_edge(
                    self.route_node(u, route_id),
                    self.walk_node(u),
                    weight=transfer_time,
                    mode="transfer"
                )
            
            network_graph.add_edge(
                    self.walk_node(v),
                    self.route_node(v, route_id),
                    weight=transfer_time,
                    mode="transfer"
                )

            network_graph.add_edge(
                self.route_node(v, route_id),
                self.walk_node(v),
                weight=transfer_time,
                mode="transfer"
            )

            route_id += 1

        return network_graph    
        
    def get_shortest_path_time(self, multimodal_graph, u, v):
        return nx.shortest_path_length(multimodal_graph, 
                                       self.walk_node(u), 
                                       self.walk_node(v), 
                                       weight="weight")
    
    def remove_duplicate_routes(self, routes):
        unique = []
        seen = set()

        for route in routes:
            sig = tuple(route['path'])  # или route_signature(route)
            if sig not in seen:
                seen.add(sig)
                unique.append(route)

        return unique
    
    def get_random_route(self):
        return random.choice(self.line_pool)
    

    # --- FITNESS EVALUATION ---
    
    def evaluate_total_time(self, network: Network):
        multimodal_graph = self.build_multimodal_graph(network)

        total = 0

        for origin in self.od_matrix.index:
            for destination, demand in self.od_matrix.loc[origin].items():
                if demand <= 0:
                    continue
                length = self.get_shortest_path_time(multimodal_graph, origin, destination)
                total += demand * length

        return total
    
    def edge_cost(self, edge_length):
        return edge_length * COST_OPERATIONAL

    def evaluate_cost(self, network: Network):
        total_cost = 0
        
        for route in network.routes:
            stops = route['path']
            for i in range(len(stops) - 1):
                u = stops[i]
                v = stops[i + 1]
                weight = self.graph.get_edge_data(u ,v)["weight"]

                total_cost = total_cost + self.edge_cost(weight)

        return total_cost
    
    def evaluate_fitness(self, network: Network) -> float:
        return (self.time_coef * self.evaluate_total_time(network)) + (self.cost_coef * self.evaluate_cost(network))


    # --- CROSSOVER ---

    def tournament_selection(self, population, k=3):
        candidates = random.sample(range(len(population)), k)
        best = min(candidates, key=lambda i: self.evaluate_fitness(population[i]))
        return population[best]

    def get_parents(self, population, k=3):
        p1 = self.tournament_selection(population, k)
        p2 = self.tournament_selection(population, k)

        while p2 is p1:
            p2 = self.tournament_selection(population, k)

        return p1, p2

    def crossover(self, parent1: Network, parent2: Network) -> Network:
        cut1 = random.randint(0, len(parent1.routes) - 1)
        cut2 = random.randint(0, len(parent2.routes) - 1)

        # print(f'from parent 1: {cut1}, from parent 2: {cut2}')

        child_routes = parent1.routes[:cut1] + parent2.routes[cut2:]
        child_routes = self.remove_duplicate_routes(child_routes)[:MAX_ROUTES]
        return Network(child_routes)
    

    # --- MUTATIONS ---

    def mutate_add(self, network: Network, p = 0.5):
        if random.random() < p and len(network.routes) < MAX_ROUTES:
            network.routes.append(self.get_random_route())
            # print('mutate add')

    def mutate_replace(self, network: Network, p = 0.5):
        if random.random() < p and len(network.routes) > 0:
            i = random.randrange(len(network.routes))
            new_route = self.get_random_route()
            network.routes[i] = new_route
            # print('mutate replace')

    def mutate_remove(self, network: Network, p = 0.1):
        if random.random() < p and len(network.routes) > 1:
            i = random.randrange(len(network.routes))
            network.routes.pop(i)
            # print('mutate remove')

    def mutate(self, network: Network):
        self.mutate_replace(network)
        self.mutate_add(network)
        self.mutate_remove(network)

        network.routes = self.remove_duplicate_routes(network.routes)


    # --- ALGORITHM ---

    def generate_initial_population(self):
        population = []
        for i in range(self.population_size):
            routes = [random.choice(self.line_pool) for i in range(self.network_size)]
            population.append(Network(routes))
        return population
    
    def get_best_solution(self, population):
        best_network = min(population, key=lambda i: self.evaluate_fitness(i))
        return best_network, self.evaluate_fitness(best_network)

    def get_elite(self, population):
        fitnesses = [self.evaluate_fitness(sol) for sol in population]

        sorted_pop = [
            sol for sol, fit in
            sorted(zip(population, fitnesses), key=lambda x: x[1])
        ]
        return sorted_pop[:self.elite_size]
    
    def generate_solution(self):
        log = loggers.GALoggerTxt('log.txt', self)
        
        population = self.generate_initial_population()
        total_best_solution, total_best_fitness = self.get_best_solution(population)
        log.log_generation('INITIAL GENERATION', population, total_best_fitness)    

        for gen in range(self.n_generations):
            gen_best_solution, gen_best_fitness = self.get_best_solution(population)
            
            log.log_generation(f'GEN {gen + 1}', population, total_best_fitness)    

            if gen_best_fitness < total_best_fitness:
                total_best_fitness = gen_best_fitness
                total_best_solution = gen_best_solution

            # new_population = []
            new_population = self.get_elite(population)

            while len(new_population) < self.population_size:
                parents = self.get_parents(population, k=3)
                child = self.crossover(parents[0], parents[1])
                self.mutate(child)
                new_population.append(child)

            population = new_population

        log.log_result(total_best_fitness)
        log.close_log()

        return total_best_solution, total_best_fitness

