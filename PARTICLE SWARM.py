import numpy as np

# Step 1: Define the function to optimize (Sphere function)
def sphere_function(x):
    return np.sum(x ** 2)

# Step 2: Initialize PSO parameters
num_particles = 30       # Number of particles in the swarm
num_dimensions = 2       # Number of dimensions of the problem
max_iterations = 100     # Max number of iterations

w = 0.5                  
c1 = 1.5                 
c2 = 2.0                 

# Search space boundaries
x_min = -10
x_max = 10
v_min = -1
v_max = 1

# Step 3: Initialize particles
positions = np.random.uniform(x_min, x_max, (num_particles, num_dimensions))
velocities = np.random.uniform(v_min, v_max, (num_particles, num_dimensions))

# Personal best positions and values
p_best_positions = positions.copy()
p_best_scores = np.apply_along_axis(sphere_function, 1, positions)

# Global best
g_best_index = np.argmin(p_best_scores)
g_best_position = p_best_positions[g_best_index]
g_best_score = p_best_scores[g_best_index]

# Step 6: Iterate
for iteration in range(max_iterations):
    
    fitness = np.apply_along_axis(sphere_function, 1, positions)

    
    better_mask = fitness < p_best_scores
    p_best_positions[better_mask] = positions[better_mask]
    p_best_scores[better_mask] = fitness[better_mask]

    
    current_g_best_index = np.argmin(p_best_scores)
    if p_best_scores[current_g_best_index] < g_best_score:
        g_best_position = p_best_positions[current_g_best_index]
        g_best_score = p_best_scores[current_g_best_index]

    
    r1 = np.random.rand(num_particles, num_dimensions)
    r2 = np.random.rand(num_particles, num_dimensions)

    cognitive = c1 * r1 * (p_best_positions - positions)
    social = c2 * r2 * (g_best_position - positions)
    velocities = w * velocities + cognitive + social

    
    velocities = np.clip(velocities, v_min, v_max)

    
    positions += velocities
    positions = np.clip(positions, x_min, x_max)

    
    if iteration % 10 == 0 or iteration == max_iterations - 1:
        print(f"Iteration {iteration}: Best Score = {g_best_score:.6f}")

# Step 7: Output the best solution
print("\n=== Optimization Complete ===")
print(f"Best Position: {g_best_position}")
print(f"Best Score: {g_best_score:.6f}")
