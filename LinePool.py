from tqdm.notebook import tqdm
import itertools

import networkx as nx
import numpy as np


def get_line_pool_mandl(graph,
                        od_matrix,
                        k_shortest_paths,
                        min_path_length,
                        max_path_length,
                        theta,
                        min_lines_per_stop):
    pool = []
    end = len(graph.nodes)

    for start_node in range(1, end):
        for end_node in range(start_node + 1, end + 1):
            shortest_paths = list(nx.shortest_simple_paths(
                graph, start_node, end_node, weight="weight"))[:k_shortest_paths]
            filtered_paths = [
                path for path in shortest_paths
                if min_path_length <=
                sum(graph[u][v]["weight"] for u, v in zip(path[:-1], path[1:]))
                <= max_path_length
            ]
            pool += filtered_paths

    pool_tmp = []
    lines_per_stop = {key: 0 for key in list(graph.nodes)}

    for path in pool:
        length = sum(graph[u][v]['weight']
                     for u, v in zip(path[:-1], path[1:]))
        demand_sum = 0
        for u in path:
            for v in path:
                if u != v:
                    demand_sum += od_matrix[u][v]
        pool_tmp.append({'path': path, 'length': length, 'demand': demand_sum})

        for stop in path:
            lines_per_stop[stop] += 1

    pool_tmp.sort(key=lambda x: -x['demand'])

    demand_threshold = np.percentile(
        [l['demand'] for l in pool_tmp], theta * 100)

    final_pool = []
    for line in pool_tmp:
        if line['demand'] >= demand_threshold:
            final_pool.append(line)
            continue

        if any(lines_per_stop[stop] <= min_lines_per_stop for stop in line['path']):
            final_pool.append(line)
        else:
            for stop in line['path']:
                lines_per_stop[stop] -= 1

    return final_pool


def _get_demand_between_connector_nodes(graph, od_matrix, from_node, to_node):
    total_demand = 0
    for from_block in graph.nodes[from_node]['blocks']:
        for to_block in graph.nodes[to_node]['blocks']:
            total_demand += od_matrix.loc[from_block, to_block]
    return total_demand


def get_line_pool_real(graph,
                       od_matrix,
                       k_shortest_paths,
                       min_path_length,
                       max_path_length,
                       theta,
                       min_lines_per_stop):
    od_pairs = (
        od_matrix
        .stack()
        .loc[lambda x: x > 1]
        .index
    )
    od_pairs = [(i, j) for i, j in od_pairs if i < j]

    pool = []
    lines_per_stop = {key: 0 for key in graph.nodes}

    for origin, destination in tqdm(od_pairs, desc="Generating routes"):

        try:
            paths_generator = nx.shortest_simple_paths(
                graph, origin, destination, weight="length_meter"
            )

            # Берём только первые k_shortest_paths
            for path in itertools.islice(paths_generator, k_shortest_paths):

                # --- длина маршрута ---
                length = sum(graph[u][v]["length_meter"]
                             for u, v in zip(path[:-1], path[1:]))

                if not (min_path_length <= length <= max_path_length):
                    continue

                # --- считаем спрос только для _connect ---
                connect_nodes = [n for n in path if str(
                    n).endswith("_connect")]

                demand_sum = 0
                for i in range(len(connect_nodes)):
                    for j in range(i + 1, len(connect_nodes)):
                        demand_sum += _get_demand_between_connector_nodes(
                            graph, od_matrix, connect_nodes[i], connect_nodes[j]
                        )

                # --- добавляем в пул ---
                pool.append({
                    "path": path,
                    "length": length,
                    "demand": demand_sum
                })

                # --- обновляем счетчики остановок ---
                for stop in path:
                    lines_per_stop[stop] += 1

        except nx.NetworkXNoPath:
            continue
        except nx.NodeNotFound:
            continue

    # --- определяем порог спроса ---
    demands = np.array([l['demand'] for l in pool])
    demand_threshold = np.percentile(demands, theta * 100)

    # --- финальная фильтрация ---
    final_pool = []

    for line in pool:

        path = line['path']
        line_demand = line['demand']

        if line_demand >= demand_threshold:
            final_pool.append(line)
            continue

        if any(lines_per_stop[stop] <= min_lines_per_stop for stop in path):
            final_pool.append(line)
            continue

        for stop in path:
            lines_per_stop[stop] -= 1

    return final_pool
