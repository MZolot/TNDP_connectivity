import GA_time_and_cost_basic


class GALoggerTxt:
    def __init__(self, file_name, GA) -> None:
        self.file = open(file_name, 'w')
        self.GA = GA

    def log_parameters(self):
        self.file.write(f"Population size: {self.GA.population_size}\n")
        self.file.write(f"Network size: {self.GA.network_size}\n")
        self.file.write(f"Number of generations: {self.GA.n_generations}\n")
        self.file.write(f"Time coefficient: {self.GA.time_coef}\n")
        self.file.write(f"Cost coefficient: {self.GA.cost_coef}\n")
        self.file.write(f"Elitism percent: {self.GA.elitism_percent}\n")

    def log_generation(self, gen_title, generation, best_fitness):
        self.file.write(f"\n{gen_title}\n")
        self.file.write(f"{'----' * len(generation)}-\n")
        for network in generation:
            self.file.write(f"| {len(network.routes)} ")
        self.file.write(f"|\n{'----' * len(generation)}-\n")
        self.file.write(f"Best fitness in generation: {best_fitness}\n\n")
        self.file.flush()

    def log_result(self, fitness):
        self.file.write(f"Total best fitness: {fitness}")

    def close_log(self):
        self.file.close()
