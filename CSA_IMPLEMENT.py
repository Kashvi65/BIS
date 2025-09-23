import random
import matplotlib.pyplot as plt
import numpy as np

# Items: (weight, value)
items = [(12, 4), (2, 2), (1, 2), (1, 1), (4, 10), (1, 2)]
max_weight = 15

n = 10          # Number of nests
pa = 0.25       # Probability to abandon a nest
max_iter = 50   # Number of iterations

def fitness(solution):
    total_weight = 0
    total_value = 0
    for i in range(len(solution)):
        if solution[i] == 1:
            total_weight += items[i][0]
            total_value += items[i][1]
    if total_weight > max_weight:
        return 0
    else:
        return total_value

def random_solution():
    return [random.randint(0,1) for _ in range(len(items))]

def levy_flight_mutation(solution):
    # Inspired by Levy flight: randomly flip between 1 to 3 bits to make bigger jumps
    new_sol = solution[:]
    flips = random.randint(1, 3)
    indices = random.sample(range(len(solution)), flips)
    for idx in indices:
        new_sol[idx] = 1 - new_sol[idx]
    return new_sol

# Initialize nests
nests = [random_solution() for _ in range(n)]

best_fitness_over_time = []
best_solution = None
best_fitness = 0

for iteration in range(max_iter):
    new_nests = []
    for nest in nests:
        # Generate new solution by levy flight mutation (bigger jumps)
        new_nest = levy_flight_mutation(nest)

        # Keep better solution
        if fitness(new_nest) > fitness(nest):
            new_nests.append(new_nest)
        else:
            new_nests.append(nest)

    # Abandon some nests randomly and replace with new random ones
    for i in range(n):
        if random.random() < pa:
            new_nests[i] = random_solution()

    nests = new_nests

    # Track best solution so far
    fitnesses = [fitness(nest) for nest in nests]
    max_fit = max(fitnesses)
    if max_fit > best_fitness:
        best_fitness = max_fit
        best_solution = nests[fitnesses.index(max_fit)]

    best_fitness_over_time.append(best_fitness)

    print(f"Iteration {iteration+1}: Best Fitness = {best_fitness}, Best Solution = {best_solution}")

print("\nFinal Best solution:", best_solution)
print("With fitness (total value):", best_fitness)

# Plot fitness progress
plt.plot(best_fitness_over_time)
plt.xlabel("Iteration")
plt.ylabel("Best Fitness")
plt.title("Fitness Improvement Over Iterations")
plt.show()
