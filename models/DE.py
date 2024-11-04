import copy
import random
import numpy as np
import json
from chromosome.index import Chromosome, SLOPE_LENGTH, SLOPE_HEIGHT, THETA


class DifferentialModel:
    def __init__(self, time_step):
        self.population_size = 25
        self.F = 0.7  # Differential weight
        self.CR = 0.8  # Crossover rate
        self.stagnation_limit = 10

        self.population = [Chromosome(time_step) for _ in range(self.population_size)]
        self.performance = []
        self.best_fitness = -np.inf
        self.best_individual = None
        self.generation = 0

    def log(self, fitness_score, override=False):
        if self.generation == 1 or self.generation % 10 == 0 or override:
            self.performance.append(
                {"generation": self.generation, "fitness": float(fitness_score)}
            )

    def train(self, max_generations=100):
        generations_without_improvement = 0

        for _ in range(max_generations):
            self.generation += 1

            next_population = []
            gen_best_fitness = -np.inf
            gen_best_individual = None

            for i in range(self.population_size):
                r1, r2, r3 = random.sample(range(self.population_size), 3)
                while r1 == i or r2 == i or r3 == i:
                    r1, r2, r3 = random.sample(range(self.population_size), 3)

                mutant = self.mutate(
                    self.population[i],
                    self.population[r1],
                    self.population[r2],
                    self.population[r3],
                )

                trial = self.crossover(self.population[i], mutant)
                next_individual, fitness = self.selection(self.population[i], trial)

                if fitness > gen_best_fitness:
                    gen_best_fitness = fitness
                    gen_best_individual = next_individual

                next_population.append(next_individual)

            self.population = next_population

            if gen_best_fitness > self.best_fitness:
                self.best_fitness = gen_best_fitness
                self.best_individual = copy.deepcopy(gen_best_individual)
                generations_without_improvement = 0
            else:
                generations_without_improvement += 1

            if generations_without_improvement >= self.stagnation_limit:
                self.log(self.best_fitness, True)
                break

            self.log(self.best_fitness)
            print(f"Generation {self.generation}")
            self.best_individual.visualize()

    def mutate(self, target, r1, r2, r3):
        mutant = copy.deepcopy(target)

        n_positions = len(mutant.magnet_positions) // 2

        for idx in range(n_positions):
            new_pos = r1.magnet_positions[idx][0] + self.F * (
                r2.magnet_positions[idx][0] - r3.magnet_positions[idx][0]
            )

            new_pos_x = max(min(new_pos, SLOPE_LENGTH), 0)

            new_pos_z = SLOPE_HEIGHT - new_pos_x * np.sin(THETA)

            mutant.magnet_positions[idx][0] = new_pos_x
            mutant.magnet_positions[idx][2] = new_pos_z
            mutant.magnet_positions[idx + n_positions][0] = new_pos_x
            mutant.magnet_positions[idx + n_positions][2] = new_pos_z

        return mutant

    def crossover(self, target, mutant):
        trial = copy.deepcopy(target)

        n_positions = len(mutant.magnet_positions) // 2

        for idx in range(n_positions):
            if random.random() < self.CR:
                trial.magnet_positions[idx][0] = mutant.magnet_positions[idx][0]
                trial.magnet_positions[idx][2] = mutant.magnet_positions[idx][2]

                trial.magnet_positions[idx + n_positions][0] = mutant.magnet_positions[
                    idx + n_positions
                ][0]
                trial.magnet_positions[idx + n_positions][2] = mutant.magnet_positions[
                    idx + n_positions
                ][2]

        return trial

    def selection(self, target, trial):
        target_fitness = target.fitness_function()
        trial_fitness = trial.fitness_function()

        if trial_fitness > target_fitness:
            return trial, trial_fitness
        else:
            return target, target_fitness

    def run(self):
        self.train()
        self.save_results()

    def save_results(self):
        with open(f"ga-bezier-curve-v1.json", "w+") as json_file:
            json.dump(
                {
                    "best_individual": self.best_individual.to_dict(),
                    "best_fitness": float(self.best_fitness),
                    "performance": self.performance,
                    "params": {
                        "population_size": self.population_size,
                        "differential_weight": self.F,
                        "crossover_rate": self.CR,
                    },
                },
                json_file,
            )
