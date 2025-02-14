# %%
from helper_tsp import *
from energy_landscape import *

from qiskit.providers.fake_provider import GenericBackendV2


ArgumentParser = argparse.ArgumentParser()
ArgumentParser.add_argument("--n", help="The size n of the problem", type=int, default=4)

# Define problem constants
n = ArgumentParser.parse_known_args()[0].n
if n<=2:
    raise ValueError("The size n of the problem must be greater than 2")

network_seed = 123
num_qubits = n**2

backend = GenericBackendV2(n)

# Create random TSP graph
graph = Tsp.create_random_instance(n, seed=network_seed).graph
distances = nx.to_numpy_array(graph)


# Draw problem's graph
draw_tsp_graph(graph)

# Brute force algorithm
classical_time, (best_distance, best_route) = tsp_brute(distances)
draw_tsp_graph(graph, best_route)
print(f"Best route = {best_route}, total distance = {best_distance}, time: {classical_time}\n")


# Quantum TSP
eigen_result, trajectory, x, ising = tsp_quantum(
    graph,
    # initial_point=[1, 0.7],
    optimizer=COBYLA(maxiter=300),
    circuit='qaoa',
    # p=2,
    # backend=backend
)
route = interpret(x)
qc = eigen_result.optimal_circuit
params = eigen_result.optimal_point
try: draw_tsp_graph(graph, route)
except Exception: pass

# Plot problem's energy field
# plot_qaoa_trajectory(trajectory, distances, qc.decompose(), tsp_cost_function, num_qubits, ising)

