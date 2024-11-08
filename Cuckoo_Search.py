import numpy as np

# Objective function: f(x, y) = x^2 + y^2
def objective_function(solution):
    x, y = solution
    return x**2 + y**2

# Lévy flight step function
def levy_flight(Lambda):
    # Step size formula for Levy flight
    sigma = (np.math.gamma(1 + Lambda) * np.sin(np.pi * Lambda / 2) /
             (np.math.gamma((1 + Lambda) / 2) * Lambda * 2**((Lambda - 1) / 2)))**(1 / Lambda)
    u = np.random.normal(0, sigma)
    v = np.random.normal(0, 1)
    step = u / abs(v)**(1 / Lambda)
    return step

# Initialize the Cuckoo Search algorithm
def cuckoo_search(n_nests=15, n_iterations=100, pa=0.25, alpha=0.01, Lambda=1.5):
    # Initialize nests (solutions)
    nests = np.random.uniform(-10, 10, (n_nests, 2))
    best_nest = nests[0]
    best_fitness = objective_function(best_nest)
    
    for iteration in range(n_iterations):
        # Generate new solutions using Levy flight
        for i in range(n_nests):
            step = levy_flight(Lambda)
            new_nest = nests[i] + alpha * step * np.random.randn(2)
            new_fitness = objective_function(new_nest)
            
            # Replace the nest if the new solution is better
            if new_fitness < objective_function(nests[i]):
                nests[i] = new_nest
            
            # Update the global best solution
            if new_fitness < best_fitness:
                best_nest = new_nest
                best_fitness = new_fitness
        

        # Abandon a fraction of the worst nests (mimicking host bird’s behavior)
        for i in range(n_nests):
            if np.random.rand() < pa:
                nests[i] = np.random.uniform(-10, 10, 2)
        
        # Update best nest if it has improved
        if objective_function(nests[i]) < best_fitness:
            best_nest = nests[i]
            best_fitness = objective_function(best_nest)
        
        print(f"Iteration {iteration + 1}, Best Fitness: {best_fitness}, Best Solution: {best_nest}")
    
    return best_nest, best_fitness

# Run the Cuckoo Search algorithm
best_solution, best_fitness = cuckoo_search()
print("\nBest Solution:", best_solution)
print("Best Fitness:", best_fitness)
