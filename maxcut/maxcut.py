# %%
from helper_maxcut import *

import random
from qiskit.visualization import plot_histogram
from qiskit.providers.fake_provider import GenericBackendV2


# Define problem constants
n = 6
seed = 123

random.seed(seed)
backend = GenericBackendV2(n)


# Create random edges array
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

# Draw problem's graph
draw_maxcut_graph(graph)


# Classical algorithms
classical_time, (brute_x, brute_cost) = maxcut_brute(weight_matrix)
print(f"Classical solution: {brute_x}, cost: {brute_cost}, time: {classical_time}")
maxcut_gw(weight_matrix)
draw_maxcut_graph(graph, brute_x, seed)


# Quantum Max-Cut
eigen_result, trajectory, x, ising = maxcut_quantum(
    weight_matrix,
    initial_point=[0.758, -0.108],
    # initial_point=[0.2, -0.2],
    optimizer=ADAM(maxiter=300),
    circuit='qaoa',
    # p=5,
    # backend=backend
)
qc = eigen_result.optimal_circuit
params = eigen_result.optimal_point
draw_maxcut_graph(graph, x, seed)

# Plot problem's energy field
plot_qaoa_trajectory(trajectory, weight_matrix, qc, maxcut_cost_function, n, ising)


# Final Circuit
# sampler = BackendSamplerV2(backend=backend)
# sampler = StatevectorSampler()
# job_result = sampler.run(([(qc.decompose(), params)]), shots=2**10).result()[0]
# counts = job_result.data.meas.get_counts()

# quant_x = max(counts, key=lambda x: counts[x])[::-1]
# print(f"Quant solution: {quant_x}, cost: {maxcut_cost_function(quant_x, weight_matrix)}, time: {eigen_result.optimizer_time}")

# plot_histogram(counts)


