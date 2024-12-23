# %%
from helper_maxcut import *

import random
from qiskit.primitives import StatevectorSampler, BackendSamplerV2
from qiskit.visualization import plot_histogram
from qiskit.providers.fake_provider import GenericBackendV2


# Define problem constants
n = 6
seed = 123

random.seed(seed)
backend = GenericBackendV2(n)


# Create adges array
edges = []
for v0 in range(n):
    for v1 in range(v0+1, n):
        w = random.randint(0,5)
        if w!=0:
            edges.append((v0, v1, w))

# Create graph
graph = nx.Graph()
graph.add_nodes_from(range(0,n))
graph.add_weighted_edges_from(edges)
weight_matrix = nx.to_numpy_array(graph)



# Brute force
start_time = time.time()
brute_x, brute_cost = maxcut_brute(weight_matrix)
brute_time = time.time() - start_time
draw_maxcut_graph(graph, brute_x, seed)
print(f"Best solution: {brute_x}, cost: {brute_cost}, time: {brute_time}")


# Quantum Max-Cut
eigen_result, trajectory, x, ising = maxcut_quantum(
    weight_matrix,
    # initial_point=[0.2, -0.2],
    initial_point=[0.758, -0.108],
    optimizer=ADAM(),
    circuit='qaoa',
#     reps=2,
#     backend=backend
)
x_best = eigen_result.best_measurement['bitstring'][::-1] #TODO: Use or delete
qc = eigen_result.optimal_circuit
params = eigen_result.optimal_point
draw_maxcut_graph(graph, x, seed)

plot_qaoa_trajectory(trajectory, weight_matrix, qc, maxcut_cost_function, n, ising)


# Final Circuit
# sampler = BackendSamplerV2(backend=backend)
# sampler = StatevectorSampler()
# job_result = sampler.run(([(qc.decompose(), params)]), shots=2**10).result()[0]
# counts = job_result.data.meas.get_counts()

# quant_x = max(counts, key=lambda x: counts[x])[::-1]
# print(f"Quant solution: {quant_x}, cost: {maxcut_cost_function(quant_x, weight_matrix)}, time: {eigen_result.optimizer_time}")

# plot_histogram(counts)


