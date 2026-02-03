import networkx as nx
import numpy as np


def get_line_pool(graph,
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
