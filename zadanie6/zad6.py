import random
import numpy as np

class Solution:

    def __init__(self, dimensions, bounds, population_size=10, max_iterations=100, fitness = "sphere"):
        self.dimensions = dimensions
        self.bounds = bounds
        self.population_size = population_size
        self.max_iterations = max_iterations
        self.population = [self.initialize_individual() for _ in range(population_size)]
        
        if fitness == 'rastrigin':
            self.fitness_function = self.rastrigin
        elif fitness == 'sphere':
            self.fitness_function = self.sphere_function

    def rastrigin(self, x):
        return 10 * len(x) + sum([(xi**2 - 10 * np.cos(2 * np.pi * xi)) for xi in x])

    def sphere_function(self, x):
        return sum(xi**2 for xi in x)
    
    def initialize_population(self, pop_size, min_val, max_val):
        return [random.uniform(min_val, max_val) for _ in range(pop_size)]
    
    def initialize_individual(self):
        return [random.uniform(self.bounds[0], self.bounds[1]) for _ in range(self.dimensions)]

    def evaluate_population(self):
        return [self.fitness_function(individual) for individual in self.population]

    def selection(self, scores):
        tournament_size = 3
        selected = random.sample(list(enumerate(scores)), tournament_size)
        selected.sort(key=lambda x: x[1])
        return self.population[selected[0][0]], self.population[selected[1][0]]

    def crossover(self, parent1, parent2):
        if random.random() < 0.7: 
            point = random.randint(1, self.dimensions - 1)
            child1 = parent1[:point] + parent2[point:]
            child2 = parent2[:point] + parent1[point:]
            return child1, child2
        return parent1, parent2

    def mutation(self, individual, mutation_rate):
        return [gene if random.random() > mutation_rate else random.uniform(self.bounds[0], self.bounds[1]) for gene in individual]

    def genetic_algorithm(self):
        mutation_rate = 0.01
        for _ in range(self.max_iterations):
            scores = self.evaluate_population()
            next_generation = []
            while len(next_generation) < self.population_size:
                parent1, parent2 = self.selection(scores)
                for child in self.crossover(parent1, parent2):
                    child = self.mutation(child, mutation_rate)
                    next_generation.append(child)
                    if len(next_generation) >= self.population_size:
                        break
            self.population = next_generation
        best_idx = scores.index(min(scores))
        return self.population[best_idx], scores[best_idx]
    
    def differential_evolution(self, F, CR):

        scores = self.evaluate_population()
        for _ in range(self.max_iterations):
            new_population = []
            for i in range(self.population_size):

                indices = list(range(self.population_size))
                indices.remove(i)
                a, b, c = random.sample(indices, 3)
                
                mutant = [self.population[a][d] + F * (self.population[b][d] - self.population[c][d]) for d in range(self.dimensions)]

                trial = [mutant[d] if random.random() < CR else self.population[i][d] for d in range(self.dimensions)]
                trial = [min(max(trial[d], self.bounds[0]), self.bounds[1]) for d in range(self.dimensions)]

                trial_fitness = self.fitness_function(trial)
                if trial_fitness < scores[i]:
                    new_population.append(trial)
                    scores[i] = trial_fitness
                else:
                    new_population.append(self.population[i])
            
            self.population = new_population
        
        best_index = scores.index(min(scores))
        return self.population[best_index], scores[best_index]
    
    def cuckoo_search(self):
        pa = 0.25  
        best_solution = min(self.population, key=self.fitness_function)
        best_fitness = self.fitness_function(best_solution)

        for _ in range(self.max_iterations):
            for i in range(self.population_size):
                new_solution = [x + random.uniform(-0.01, 0.01) for x in self.population[i]]
                new_fitness = self.fitness_function(new_solution)
                if new_fitness < self.fitness_function(self.population[i]):
                    self.population[i] = new_solution
                    if new_fitness < best_fitness:
                        best_solution, best_fitness = new_solution, new_fitness

            for i in range(self.population_size):
                if random.random() < pa:
                    self.population[i] = self.initialize_individual()

        return best_solution, best_fitness

    def particle_swarm(self):

        velocity = [[0] * self.dimensions for _ in range(self.population_size)]
        personal_best = self.population[:]
        personal_best_scores = self.evaluate_population()
        global_best = min(personal_best, key=self.fitness_function)
        global_best_score = self.fitness_function(global_best)

        for _ in range(self.max_iterations):
            for i in range(self.population_size):
                r1, r2 = random.random(), random.random()
                velocity[i] = [0.5 * v + 0.8 * r1 * (pb - p) + 0.9 * r2 * (gb - p)
                               for v, p, pb, gb in zip(velocity[i], self.population[i], personal_best[i], global_best)]

                self.population[i] = [max(min(p + v, self.bounds[1]), self.bounds[0]) 
                                      for p, v in zip(self.population[i], velocity[i])]

                current_fitness = self.fitness_function(self.population[i])

                if current_fitness < personal_best_scores[i]:
                    personal_best[i] = self.population[i]
                    personal_best_scores[i] = current_fitness

                    if current_fitness < global_best_score:
                        global_best = self.population[i]
                        global_best_score = current_fitness

        return global_best, global_best_score

    def bee_algorithm(self):
        best_solution = min(self.population, key=self.fitness_function)
        best_fitness = self.fitness_function(best_solution)

        for _ in range(self.max_iterations):
            for i in range(self.population_size):
                new_solution = self.population[i] + np.random.uniform(-1, 1, self.dimensions)
                new_solution = np.clip(new_solution, self.bounds[0], self.bounds[1])
                if self.fitness_function(new_solution) < self.fitness_function(self.population[i]):
                    self.population[i] = new_solution
                    if self.fitness_function(new_solution) < best_fitness:
                        best_solution, best_fitness = new_solution, self.fitness_function(new_solution)
                        
            for i in range(self.population_size // 2):
                if random.random() < 0.1: 
                    self.population[i] = self.initialize_individual()
        return best_solution, best_fitness

    def ant_algorithm(self):
        path_cost = np.inf
        best_path = None
        pheromone_levels = np.ones((self.dimensions, self.dimensions))

        for _ in range(self.max_iterations):
            paths = []
            for _ in range(self.population_size):
                path = list(range(self.dimensions))
                random.shuffle(path)
                paths.append(path)
            
            for path in paths:
                cost = self.path_cost(path)
                if cost < path_cost:
                    path_cost = cost
                    best_path = path
                for i in range(len(path) - 1):
                    pheromone_levels[path[i]][path[i + 1]] += 1 / cost
        
        return best_path, path_cost

    def bat_algorithm(self):
        velocity = [[0] * self.dimensions for _ in range(self.population_size)]
        freq_min, freq_max = 0, 2
        loudness = [1] * self.population_size
        pulse_rate = [0.1] * self.population_size
        fitness = self.evaluate_population()

        best_idx = fitness.index(min(fitness))
        best_solution = self.population[best_idx]
        best_fitness = fitness[best_idx]

        for _ in range(self.max_iterations):
            for i in range(self.population_size):

                freq = freq_min + (freq_max - freq_min) * random.random()

                velocity[i] = [v + (p - best_solution[j]) * freq for j, (v, p) in enumerate(zip(velocity[i], self.population[i]))]
                new_position = [p + v for p, v in zip(self.population[i], velocity[i])]
                new_position = [max(min(p, self.bounds[1]), self.bounds[0]) for p in new_position]  # Apply bounds

                if random.random() < pulse_rate[i] or self.fitness_function(new_position) < best_fitness:
                    new_fitness = self.fitness_function(new_position)
                    if new_fitness < fitness[i] or random.random() < loudness[i]:
                        self.population[i] = new_position
                        fitness[i] = new_fitness
                        loudness[i] *= 0.9
                        pulse_rate[i] = 0.1  
                        
                        if new_fitness < best_fitness:
                            best_solution = new_position
                            best_fitness = new_fitness

        return best_solution, best_fitness

    
    def firefly_algorithm(self):

        light_absorption = 0.5
        fitness = self.evaluate_population()
        for _ in range(self.max_iterations):
            for i in range(self.population_size):
                for j in range(self.population_size):
                    if fitness[j] < fitness[i]: 
                        attractiveness = np.exp(-light_absorption * sum([(self.population[i][d] - self.population[j][d])**2 for d in range(self.dimensions)]))
                        self.population[i] = [self.population[i][d] + attractiveness * (self.population[j][d] - self.population[i][d]) for d in range(self.dimensions)]
                        self.population[i] = [max(min(self.population[i][d], self.bounds[1]), self.bounds[0]) for d in range(self.dimensions)]
                        new_fitness = self.fitness_function(self.population[i])
                        if new_fitness < fitness[i]:
                            fitness[i] = new_fitness 

        best_index = fitness.index(min(fitness))
        return self.population[best_index], fitness[best_index]

if __name__ == "__main__":
    dimensions = 10
    bounds = (-5.12, 5.12)
    population_size = 50
    max_iterations = 100
    mutation_rate = 0.1
    F = 0.8 
    CR = 0.9  

    rastrigin_solution = Solution(dimensions, bounds, population_size, max_iterations, 'rastrigin')
    sphere_solution = Solution(dimensions, bounds, population_size, max_iterations, 'sphere')

    print("Testing algorithms with Rastrigin function:")
    _, rastrigin_fitness = rastrigin_solution.genetic_algorithm()
    print(f"Genetic Algorithm (Rastrigin): Best Fitness = {rastrigin_fitness}")

    _, rastrigin_fitness = rastrigin_solution.differential_evolution(F, CR)
    print(f"Differential Evolution (Rastrigin): Best Fitness = {rastrigin_fitness}")

    _, rastrigin_fitness = rastrigin_solution.particle_swarm()
    print(f"Particle Swarm (Rastrigin): Best Fitness = {rastrigin_fitness}")

    _, rastrigin_fitness = rastrigin_solution.bee_algorithm()
    print(f"Bee Algorithm (Rastrigin): Best Fitness = {rastrigin_fitness}")

    _, rastrigin_fitness = rastrigin_solution.bat_algorithm()
    print(f"Bat Algorithm (Rastrigin): Best Fitness = {rastrigin_fitness}")

    _, rastrigin_fitness = rastrigin_solution.firefly_algorithm()
    print(f"Firefly Algorithm (Rastrigin): Best Fitness = {rastrigin_fitness}")

    print("\nTesting algorithms with Sphere function:")
    _, sphere_fitness = sphere_solution.genetic_algorithm()
    print(f"Genetic Algorithm (Sphere): Best Fitness = {sphere_fitness}")

    _, sphere_fitness = sphere_solution.differential_evolution(F, CR)
    print(f"Differential Evolution (Sphere): Best Fitness = {sphere_fitness}")

    _, sphere_fitness = sphere_solution.particle_swarm()
    print(f"Particle Swarm (Sphere): Best Fitness = {sphere_fitness}")

    _, sphere_fitness = sphere_solution.bee_algorithm()
    print(f"Bee Algorithm (Sphere): Best Fitness = {sphere_fitness}")

    _, sphere_fitness = sphere_solution.bat_algorithm()
    print(f"Bat Algorithm (Sphere): Best Fitness = {sphere_fitness}")

    _, sphere_fitness = sphere_solution.firefly_algorithm()
    print(f"Firefly Algorithm (Sphere): Best Fitness = {sphere_fitness}")