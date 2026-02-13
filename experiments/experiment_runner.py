import Mandl
import LinePool
import TNDP
import GA_basic

import csv
import time

mandl_network = Mandl.MandlNetwork()
mandl_graph = mandl_network.graph
mandl_pedestrian = mandl_network.graph_pedestrian
mandl_od = mandl_network.od_matrix

line_pool = LinePool.get_line_pool(mandl_graph, mandl_od, 10, 16, 59, 0.7, 2)


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


filename = "experiments.csv"

fieldnames = [
    "run_id",
    "tested_parameter",
    "initial_population_size",
    "initial_network_size",
    "max_network_size",
    "generations",
    # "mutation_rate",
    # "crossover_rate",
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

def write_result(run_id, tested_parameter, params, result, filename=filename):
    row = {
        "run_id": run_id,
        "tested_parameter": tested_parameter,
        **params,
        **result
    }

    with open(filename, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        writer.writerow(row)
    
iterations = 10 

for tested_parameter in experiment_params.keys():
    current_params = {}
    for key in experiment_params.keys():
        current_params[key] = experiment_params[key].default_value

    tested_value = experiment_params[tested_parameter].start_value
    max_value = experiment_params[tested_parameter].max_value
    iteration = experiment_params[tested_parameter].iteration_delta
    while tested_value <= max_value:
        print(f'Testing {tested_parameter}, {tested_value}/{max_value}')
        for i in range(iterations):
            current_params[tested_parameter] = tested_value
            tndp = TNDP.TNDP(mandl_graph,
                            mandl_pedestrian,
                            mandl_od,
                            line_pool,
                            max_network_size=current_params['max_network_size'],
                            time_weight=1,
                            cost_weight=0.2,
                            connectivity_weight=10000)
            
            
            start = time.perf_counter()

            ga = GA_basic.GeneticAlgorithm(tndp,
                                        initial_population_size=current_params['initial_population_size'],
                                        initial_population_network_size=current_params['initial_network_size'],
                                        n_generations=current_params['generations'])
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
            write_result(i + 1, tested_parameter, current_params, res)

        tested_value += iteration
