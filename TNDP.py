import networkx as nx


class Network:
    def __init__(self, routes):
        self.routes = routes
        self.total_fintess: float = -1 
        self.objective_fitnesses = {}
        self.multimodal_graph: nx.Graph | None = None


class TNDP:
    COST_OPERATIONAL = 10000

    def __init__(self,
                 graph,
                 pedestrian_graph,
                 od_matrix,
                 line_pool,
                 initial_network_size=10,
                 max_network_size=20,
                 time_weight=10.0,
                 cost_weight=1.0) -> None:

        self.graph = graph
        self.pedestrian_graph = pedestrian_graph
        self.od_matrix = od_matrix
        self.line_pool = line_pool
        self.initial_network_size = initial_network_size
        self.max_network_size = max_network_size
        self.time_weight = time_weight
        self.cost_weight = cost_weight

    def walk_node(self, v):
        return (v, None)

    def route_node(self, v, route_id):
        return (self, v, route_id)

    def build_multimodal_graph(self, network: Network):
        if network.multimodal_graph is not None:
            return network.multimodal_graph

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
                weight = self.graph.get_edge_data(u, v)["weight"]

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

        network.multimodal_graph = network_graph

        return network_graph

    def get_shortest_path_time(self, multimodal_graph, u, v):
        return nx.shortest_path_length(multimodal_graph,
                                       self.walk_node(u),
                                       self.walk_node(v),
                                       weight="weight")

    def evaluate_total_time(self, network: Network):
        if 'time' in network.objective_fitnesses:
            return network.objective_fitnesses['time']
        
        multimodal_graph = self.build_multimodal_graph(network)

        total = 0

        for origin in self.od_matrix.index:
            for destination, demand in self.od_matrix.loc[origin].items():
                if demand <= 0:
                    continue
                length = self.get_shortest_path_time(
                    multimodal_graph, origin, destination)
                total += demand * length

        return total

    def edge_cost(self, edge_length):
        return edge_length * self.COST_OPERATIONAL

    def evaluate_cost(self, network: Network):
        if 'cost' in network.objective_fitnesses:
            return network.objective_fitnesses['cost']
        
        total_cost = 0

        for route in network.routes:
            stops = route['path']
            for i in range(len(stops) - 1):
                u = stops[i]
                v = stops[i + 1]
                weight = self.graph.get_edge_data(u, v)["weight"]

                total_cost = total_cost + self.edge_cost(weight)

        return total_cost

    def evaluate_fitness(self, network: Network) -> float:
        if network.total_fintess != -1:
            return network.total_fintess
        return (self.time_weight * self.evaluate_total_time(network)) + (self.cost_weight * self.evaluate_cost(network))
