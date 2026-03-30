import os
from datetime import datetime
import json
import pickle

from ga.TNDP import TndpNetwork, TNDP
from ga.GA_basic import GeneticAlgorithm


# === ALGORITHM INPUT ===

def save_graph(graph, filepath: str):
    with open(filepath, "wb") as f:
        pickle.dump(graph, f)


def load_graph(filepath: str):
    with open(filepath, "rb") as f:
        graph = pickle.load(f)
    return graph


def save_od_matrix(df, filepath: str):
    df.to_parquet(filepath)


def load_od_matrix(filepath: str):
    import pandas as pd
    return pd.read_parquet(filepath)


def save_line_pool(line_pool, filepath: str):
    data = {
        "lines": line_pool,
        "meta": {
            "num_lines": len(line_pool)
        }
    }

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def load_line_pool(filepath: str):
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["lines"]


def save_all_data(graphs: dict,
                  od_matrix,
                  line_pool,
                  folder: str = "data"):
    """
    graphs: dict вида {
        "graph": G,
        "pedestrian_graph": G_walk,
        "graph2": ...,
        ...
    }
    """

    os.makedirs(folder, exist_ok=True)

    for name, graph in graphs.items():
        path = os.path.join(folder, f"{name}.pkl")
        with open(path, "wb") as f:
            pickle.dump(graph, f, protocol=pickle.HIGHEST_PROTOCOL)

    od_path = os.path.join(folder, "od_matrix.parquet")
    od_matrix.to_parquet(od_path)

    clean_line_pool = [[int(v) for v in route] for route in line_pool]

    lp_path = os.path.join(folder, "line_pool.json")
    with open(lp_path, "w", encoding="utf-8") as f:
        json.dump({
            "lines": clean_line_pool,
            "meta": {"num_lines": len(clean_line_pool)}
        }, f, ensure_ascii=False, indent=4)


# === ALGORITHM OUTPUT ===


def save_network(network: TndpNetwork, filepath: str):
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(network.to_dict(), f, ensure_ascii=False, indent=4)


def load_network(filepath: str) -> TndpNetwork:
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    return TndpNetwork.from_dict(data)


def save_TNDP(tndp: TNDP, filepath: str):
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(tndp.to_dict(), f, ensure_ascii=False, indent=4)


def load_TNDP(filepath: str,
              graph,
              pedestrian_graph,
              od_matrix,
              line_pool):

    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    tndp = TNDP.from_dict(
        data,
        graph,
        pedestrian_graph,
        od_matrix,
        line_pool
    )

    return tndp


def save_ga(ga, filepath: str):
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(ga.to_dict(), f, ensure_ascii=False, indent=4)


def load_ga(filepath: str, tndp: TNDP):

    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    ga = GeneticAlgorithm.from_dict(
        data,
        tndp
    )

    return ga


def save_experiment(network, tndp, ga, filepath: str):
    data = {
        "tndp_params": tndp.to_dict(),
        "genetic_algorithm": ga.to_dict(),
        "network": network.to_dict(),

        "meta": {
            "timestamp": datetime.now().isoformat()
        }
    }

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def load_experiment(filepath,
                    graph,
                    pedestrian_graph,
                    od_matrix,
                    line_pool):

    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    network = TndpNetwork.from_dict(data["network"])

    tndp = TNDP.from_dict(
        data["tndp_params"],
        graph,
        pedestrian_graph,
        od_matrix,
        line_pool
    )

    ga = GeneticAlgorithm.from_dict(
        data["genetic_algorithm"],
        tndp
    )

    return network, tndp, ga
