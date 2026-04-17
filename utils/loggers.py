import time


class GALoggerTxt:
    def __init__(self, file_name, GA) -> None:
        self.file_name = file_name
        self.GA = GA
        self.start_time = time.time()

    def log_parameters(self):
        with open(self.file_name, 'w', encoding='utf-8') as f:
            f.write("GA Parameters:\n")
            ga_params = self.GA.to_dict()
            for key in ga_params.keys():
                f.write(f"  {key}: {ga_params[key]}\n")

            f.write("\nTNDP Parameters:\n")
            tndp_params = self.GA.tndp.to_dict()
            for key in tndp_params.keys():
                f.write(f"  {key}: {tndp_params[key]}\n")
            f.write(
                '\n==============================================================\n')
            f.flush()

    def log_generation(self, gen_title, generation, best_fitness):
        with open(self.file_name, 'a', encoding='utf-8') as f:
            f.write(f"\n{gen_title}\n")
            f.write(f"{'-----' * len(generation)}-\n")
            for network in generation:
                f.write(f"| {len(network.routes)} ")
            f.write(f"|\n{'-----' * len(generation)}-\n")
            f.write(f"Best fitness in generation: {best_fitness}\n\n")
            f.flush()

    def log_result(self, fitness, solution_gen, solution):
        with open(self.file_name, 'a', encoding='utf-8') as f:
            f.write(
                '==============================================================\n\n')
            f.write(f"Total best fitness: {fitness:.3f}\n")
            f.write(f"Achieved on gen: {solution_gen}\n\n")
            f.write(
                f"Execution time: {format_time(time.time() - self.start_time)}\n\n")

            f.write(f"Objective fitnesses:\n")
            for key in solution.objective_fitnesses.keys():
                f.write(f"  {key}: {solution.objective_fitnesses[key]:.3f}\n")
            f.flush()


def format_time(seconds):
    hours = int(seconds // 3600)
    mins = int(seconds // 60)
    secs = seconds % 60
    return f"{hours} h {mins} min {secs:.2f} sec"
