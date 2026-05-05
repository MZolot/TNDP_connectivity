import src.Mandl as Mandl
import src.ga.LinePool as lp
import src.ga.TNDP as TNDP
import src.ga.GA_basic as GA

import csv
import time

mandl_network = Mandl.MandlNetwork()
mandl_graph = mandl_network.graph
mandl_pedestrian = mandl_network.graph_pedestrian
mandl_od = mandl_network.od_matrix

line_pool = lp.get_line_pool_mandl(mandl_graph, mandl_od, 10, 16, 59, 0.7, 2)

class Parameter():
    def __init__(self, start_value, max_value, default_value, iteration_delta) -> None:
        self.start_value = start_value
        self.max_value = max_value
        self.default_value = default_value
        self.iteration_delta = iteration_delta


experiment_params = {
    'initial_population_size': Parameter(5, 50, 35, 5),
    'initial_network_size': Parameter(5, 50, 15, 5),
    'max_network_size': Parameter(5, 50, 20, 5),
    'generations': Parameter(5, 50, 45, 5)
}

weights_list = [
    (1, 100, 1),
    (1, 10, 1),
    (1, 5, 1),
    (1, 2, 1),
    (1, 1.5, 1),
    (1, 1, 1),
    (1, 0.1, 1),
    (1, 0, 1),
    (1, 0.01, 1),
    (10, 1, 1),
    (1, 1, 10)
]


filename = "weights_experiments_full.csv"

fieldnames = [
    # "run_id",
    "time_weight",
    "cost_weight",
    "connectivity_weight",
    # "initial_population_size",
    # "initial_network_size",
    # "max_network_size",
    # "generations",
    "time",
    "cost",
    "connectivity",
    "fitness",
    "best_fitness_gen",
    "solution_size",
    "execution_time"
]

with open(filename, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()


def write_result(run_id, weights, result, filename=filename):
    row = {
        "run_id": run_id,
        **weights,
        # **params,
        **result
    }

    with open(filename, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        writer.writerow(row)


iterations = 20

current_params = {
    key: experiment_params[key].default_value
    for key in experiment_params.keys()
}

run_id = 1

for weights in weights_list:
    time_w, cost_w, conn_w = weights
    
    print(f"Testing weights: {weights}")
    
    for i in range(iterations):
        print(run_id, end=' ')
        run_id += 1
        
        tndp = TNDP.TNDP(
            mandl_graph,
            mandl_pedestrian,
            mandl_od,
            line_pool,
            max_network_size=current_params['max_network_size'],
            time_weight=time_w,
            cost_weight=cost_w,
            connectivity_weight=conn_w * 100000,
            mandl=True
        )

        start = time.perf_counter()

        ga = GA.GeneticAlgorithm(
            tndp,
            population_size=current_params['initial_population_size'],
            initial_network_size=current_params['initial_network_size'],
            n_generations=current_params['generations']
        )

        solution, fitness, gen = ga.generate_solution()

        end = time.perf_counter()
        exec_time = end - start

        res = {
            "time": solution.objective_fitnesses['time'],
            "cost": solution.objective_fitnesses['cost'],
            "connectivity": round(solution.objective_fitnesses['connectivity'], 4),
            "fitness": round(fitness, 4),
            "best_fitness_gen": gen,
            "solution_size": len(solution.routes),
            "execution_time": round(exec_time, 4)
        }

        weights_dict = {
            "time_weight": time_w,
            "cost_weight": cost_w,
            "connectivity_weight": conn_w
        }

        # write_result(run_id, current_params, weights_dict, res)
        write_result(run_id, weights_dict, res)
        
    print()
