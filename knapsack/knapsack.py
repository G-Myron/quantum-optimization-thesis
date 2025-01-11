#%%
from helper_knapsack import *

import random
from qiskit.primitives import StatevectorSampler, BackendSamplerV2
from qiskit.visualization import plot_histogram
from qiskit.providers.fake_provider import GenericBackendV2


# Define problem constants
n = 3
seed = 123

random.seed(seed)
backend = GenericBackendV2(10)


# Create random profits and weights arrays
max_weight = 50
profits = [random.randint(1, 50) for _ in range(n)]
weights = [random.randint(1, 50) for _ in range(n)]



# Classical dynamic programming
classical_time, (classical, items, result) = knapsack_dynamic(max_weight, weights, profits)
print(f"Best value: {classical}, time: {classical_time} \nItems (pos,weight,value): {items} \nResult: {result}\n")


# Quantum Knapsack
eigen_result, trajectory, x, ising, converter = knapsack_quantum(
    profits, weights, max_weight,
    initial_point=[1, -0.1],
    optimizer=ADAM(maxiter=300),
    # reps=5,
    # backend=backend
)
qc = eigen_result.optimal_circuit
params = eigen_result.optimal_point

# Plot problem's energy field
# plot_qaoa_trajectory(trajectory, profits, qc.decompose(), make_knapsack_cost_function(converter), qc.num_qubits, ising)


# Final Circuit
# sampler = BackendSamplerV2(backend=backend)
# sampler = StatevectorSampler()
# job_result = sampler.run(([(qc.decompose().decompose(), params)])).result()[0]
# counts = job_result.data.meas.get_counts()

# plot_histogram({i:counts[i] for i in counts if counts[i]>0.01*2**9})
# plot_histogram({''.join([str(int(i)) for i in s.x]): s.probability for s in samples})
# plot_histogram({''.join([str(int(i)) for i in s.x]): s.probability for s in raw_samples if s.probability > 0.005})

