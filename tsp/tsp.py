# %%
from helper_tsp import *
from energy_landscape import *

import random
import time



# Define problem constants
n = 3
network_seed = 123
num_qubits = n**2

# Create random TSP graph
graph = Tsp.create_random_instance(n, seed=network_seed).graph
distances = nx.to_numpy_array(graph)



# Draw problem's graph
draw_tsp_graph(graph)

# Brute force algorithm
classical_time, (best_distance, best_route) = tsp_brute(distances)
draw_tsp_graph(graph, best_route)
print(f"Best route = {best_route}, total distance = {best_distance}, time: {classical_time}")


# Quantum TSP
eigen_result, trajectory, x, ising = tsp_quantum(
    graph,
    # initial_point=[1, 0.7],
    # optimizer=ADAM(maxiter=300),
    # circuit='qaoa',
    # reps=5,
    # backend=backend
)
route = interpret(x)
qc = eigen_result.optimal_circuit
params = eigen_result.optimal_point
try: draw_tsp_graph(graph, route)
except Exception: pass

# Plot problem's energy field
# plot_qaoa_trajectory(trajectory, distances, qc.decompose(), tsp_cost_function, num_qubits, ising)

