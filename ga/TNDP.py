import math

import numpy as np
import networkx as nx
import pandas as pd


BUS_SPEED_KM_H = 25
HUMAN_SPEED_KM_H = 5

BUS_SPEED_M_MIN = (BUS_SPEED_KM_H * 1000) / 60
HUMAN_SPEED_M_MIN = (HUMAN_SPEED_KM_H * 1000) / 60


class TndpNetwork:
    def __init__(self, routes):
        self.routes = routes
        self.total_fintess: float = -1
        self.objective_fitnesses = {}
        self.multimodal_graph: nx.Graph | None = None

    def __len__(self) -> int:
        return len(self.routes)

    def __str__(self) -> str:
        return f'TNDP network with {len(self.routes)}, weighted fitness: {self.total_fintess}'

    def __repr__(self) -> str:
        tab = '             '
        routes = f'Routes={self.__len__()}'
        fitness = f'weighted fitness: {self.total_fintess:.0f}'
        objectives = [
            f'{i}: {self.objective_fitnesses[i]:.3f}' for i in self.objective_fitnesses.keys()]
        return f'TNDP Network({routes};\n{tab}{fitness};\n{tab}{objectives})'


class TNDP:
    COST_OPERATIONAL = 1000

    def __init__(self,
                 graph,
                 pedestrian_graph,
                 od_matrix,
                 line_pool,
                 max_network_size=20,
                 time_weight=1.0,
                 cost_weight=1.0,
                 connectivity_weight=10000.0,
                 mandl=False) -> None:

        self.graph = graph
        self.pedestrian_graph = pedestrian_graph
        self.od_matrix = od_matrix
        self.line_pool = line_pool
        self.max_network_size = max_network_size
        self.time_weight = time_weight
        self.cost_weight = cost_weight
        self.connectivity_weight = connectivity_weight
        self.mandl = mandl

        if mandl:
            self._shortest_times = self._calculate_shortest_possible_times_mandl(
                self.graph)
        else:
            self._shortest_times = self._calculate_shortest_possible_times_real(
                self.pedestrian_graph)

    def _calculate_shortest_possible_times_mandl(self, graph):
        n = graph.number_of_nodes()
        dist_matrix = np.full((n + 1, n + 1), np.inf)

        nodes = sorted(graph.nodes())

        for i, source in enumerate(nodes):
            lengths = nx.single_source_dijkstra_path_length(
                graph,
                source,
                weight='weight'
            )

            for j, target in enumerate(nodes):
                if target in lengths:
                    dist_matrix[i + 1, j + 1] = lengths[target]

        return dist_matrix

    def _calculate_shortest_possible_times_real(self, graph):
        centroids = [n for n, data in graph.nodes(
            data=True) if data.get('type') == 'centroid']

        def travel_time(u, v, data):
            if data.get("on_street"):
                length = data.get("length", 0)
                return length / BUS_SPEED_M_MIN
            else:
                return data["weight"]

        travel_times = {}

        for c in centroids:
            lengths = nx.single_source_dijkstra_path_length(
                graph, c, weight=travel_time)
            travel_times[c] = {t: lengths[t]
                               for t in centroids if t in lengths and t != c}

        return travel_times

    def walk_node(self, v):
        return (v, None)

    def route_node(self, v, route_id):
        return (v, route_id)

    def build_multimodal_graph(self, network: TndpNetwork):
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
            for i in range(len(route) - 1):
                u = route[i]
                v = route[i + 1]
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

    def get_shortest_path_time_in_multimodal(self, multimodal_graph, u, v):
        return nx.shortest_path_length(multimodal_graph,
                                       self.walk_node(u),
                                       self.walk_node(v),
                                       weight="weight")

    def evaluate_total_time(self, network: TndpNetwork):
        if 'time' in network.objective_fitnesses:
            return network.objective_fitnesses['time']

        multimodal_graph = self.build_multimodal_graph(network)

        total_time = 0

        for origin in self.od_matrix.index:
            if self.walk_node(origin) not in multimodal_graph:
                continue
            for destination, demand in self.od_matrix.loc[origin].items():
                if demand <= 0:
                    continue
                if self.walk_node(destination) not in multimodal_graph:
                    continue
                length = self.get_shortest_path_time_in_multimodal(
                    multimodal_graph, origin, destination)
                total_time += demand * length

        network.objective_fitnesses['time'] = total_time

        return total_time

    def edge_cost(self, edge_length):
        return edge_length * self.COST_OPERATIONAL

    def evaluate_cost(self, network: TndpNetwork):
        if 'cost' in network.objective_fitnesses:
            return network.objective_fitnesses['cost']

        total_cost = 0

        for route in network.routes:
            for i in range(len(route) - 1):
                u = route[i]
                v = route[i + 1]
                weight = self.graph.get_edge_data(u, v)["weight"]

                total_cost = total_cost + self.edge_cost(weight)

        network.objective_fitnesses['cost'] = total_cost

        return total_cost

    def evaluate_connectivity_mandl(self, network: TndpNetwork):
        if 'connectivity' in network.objective_fitnesses:
            return network.objective_fitnesses['connectivity']

        if network.multimodal_graph is None:
            mm_graph = self.build_multimodal_graph(network)
        else:
            mm_graph = network.multimodal_graph

        total_connectivity = 0

        n = self.graph.number_of_nodes()
        for i in range(1, n + 1):
            for j in range(i + 1, n + 1):
                base_time = self._shortest_times[i, j]
                network_time = self.get_shortest_path_time_in_multimodal(
                    mm_graph, i, j)
                total_connectivity += network_time / base_time

        p = (n * (n - 1)) / 2

        network.objective_fitnesses['connectivity'] = total_connectivity / p

        return total_connectivity / p

    def evaluate_connectivity_real(self, network: TndpNetwork):
        if 'connectivity' in network.objective_fitnesses:
            return network.objective_fitnesses['connectivity']

        if network.multimodal_graph is None:
            mm_graph = self.build_multimodal_graph(network)
        else:
            mm_graph = network.multimodal_graph

        total_connectivity = 0
        for origin in self.od_matrix.index:
            if self.walk_node(origin) not in mm_graph:
                continue

            for destination in self.od_matrix.loc[origin].index:
                if self.walk_node(destination) not in mm_graph or origin == destination:
                    continue

                base_time = self._shortest_times[origin][destination]
                if not isinstance(base_time, (int, float)) or math.isnan(base_time) or base_time == 0:
                    continue

                network_time = self.get_shortest_path_time_in_multimodal(
                    mm_graph, origin, destination)
                total_connectivity += network_time / base_time

        p = sum(
            1
            for origin in self.od_matrix.index if self.walk_node(origin) in mm_graph
            for destination in self.od_matrix.loc[origin].index
            if self.walk_node(destination) in mm_graph and origin != destination
        )

        network.objective_fitnesses['connectivity'] = total_connectivity / p

        return total_connectivity / p

    def evaluate_fitness(self, network: TndpNetwork) -> float:
        if network.total_fintess != -1:
            return network.total_fintess

        weighted_time_fitness = self.time_weight * \
            self.evaluate_total_time(network)
        weighted_cost_fitness = self.cost_weight * self.evaluate_cost(network)

        if self.mandl:
            weighted_connectivity_fitness = self.connectivity_weight * \
                self.evaluate_connectivity_mandl(network)
        else:
            weighted_connectivity_fitness = self.connectivity_weight * \
                self.evaluate_connectivity_real(network)

        total_fintess = weighted_time_fitness + \
            weighted_cost_fitness + weighted_connectivity_fitness

        network.total_fintess = total_fintess

        return total_fintess

    def get_connectivity_for_node(self, multimodal_graph, node_id):
        number_of_nodes = len(self.graph.nodes)
        total_connectivity = 0
        for j in range(1, number_of_nodes + 1):
            if j == node_id:
                continue
            base_time = self._shortest_times[node_id, j]
            network_time = self.get_shortest_path_time_in_multimodal(
                multimodal_graph, node_id, j)
            total_connectivity += network_time / base_time
        return total_connectivity / (number_of_nodes - 1)
