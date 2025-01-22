from sys import path
path.append('..')
from energy_landscape import *
from problem_quantum import *

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

from qiskit_optimization import QuadraticProgram
from qiskit_optimization.algorithms import GoemansWilliamsonOptimizer
from qiskit_algorithms import SamplingVQE
from qiskit_algorithms.optimizers import SPSA, ADAM, COBYLA, SLSQP
from qiskit.circuit.library.n_local.qaoa_ansatz import QAOAAnsatz
from qiskit.circuit.library import TwoLocal
from qiskit.primitives import Sampler, StatevectorSampler, BackendSampler, BackendSamplerV2


# Constants
default_color = '#1f78b4'
seed = 123

# Helper functions
def draw_maxcut_graph(graph, bitstring=None, network_seed=seed):
    if not bitstring: colors = default_color
    else: colors = [default_color if int(n)==0 else "r" for n in bitstring]

    pos = nx.spring_layout(graph, seed=network_seed)
    nx.draw_networkx(graph, pos, node_color=colors, node_size=600)

    edge_labels = nx.get_edge_attributes(graph, "weight")
    nx.draw_networkx_edge_labels(graph, pos, edge_labels)
    plt.show()


def maxcut_cost_function(bitstring, weight_matrix):
    n = weight_matrix.shape[0]
    cost = 0
    for i in range(n):
        for j in range(i+1, n):
            if bitstring[i] != bitstring[j]:
                cost += weight_matrix[i,j]
    return cost

@time_func
def maxcut_brute(weight_matrix):
    n = weight_matrix.shape[0]
    best_cost = 0
    for b in range(2**n):
        x = bin(b)[2:].zfill(n)[::-1]
        cost = maxcut_cost_function(x, weight_matrix)

        if best_cost < cost:
            best_cost, best_x = cost, x
    
    return best_x, best_cost

def maxcut_gw(weight_matrix):
    n = weight_matrix.shape[0]

    qp = QuadraticProgram("Max-Cut")
    qp.binary_var_list(n, "x")
    qp.maximize(
        quadratic=-weight_matrix,
        linear=sum(row for row in weight_matrix)
    )

    GWOptimizer = GoemansWilliamsonOptimizer(num_cuts=1)
    result = GWOptimizer.solve(qp)
    print(result.prettyprint())


def maxcut_quantum(weight_matrix:np.ndarray, *,
                   optimizer=SLSQP(maxiter=300), circuit='', initial_point=None, p=1, backend=None):
    n = weight_matrix.shape[0]
    sampler = BackendSampler(backend=backend) if backend else Sampler()
    # In the new version:
    # sampler = BackendSamplerV2(backend=backend) if backend else StatevectorSampler()

    # Define the Quadratic Program for the problem
    qp = QuadraticProgram("Max-Cut")
    qp.binary_var_list(n, "x")

    qubo_matrix = -weight_matrix
    qubo_vector = sum(row for row in weight_matrix)

    qp.maximize(quadratic=qubo_matrix, linear=qubo_vector)

    # Map to the Ising model
    operator, offset = qp.to_ising()

    # VQE callback function to save the trajectory of the optimization parameters
    trajectory={'beta_0':[], 'gamma_0':[], 'energy':[]}
    def callback(eval_count, params, value, std_dev):
        trajectory['beta_0'].append(params[1])
        trajectory['gamma_0'].append(params[0])
        trajectory['energy'].append( -value -offset)

    # Choose the quantum circuit
    if circuit.lower()=='qaoa':
        ansatz = QAOAAnsatz(operator, reps=p).decompose()
    else:
        ansatz = TwoLocal(operator.num_qubits, "ry", "cz", entanglement="linear", reps=p)

    # Create and run the VQE
    vqe = SamplingVQE(sampler, ansatz, optimizer, initial_point=initial_point, callback=callback)
    eigen_result = vqe.compute_minimum_eigenvalue(operator)

    # Compute the most likely output
    probabilities = eigen_result.eigenstate.binary_probabilities()
    x_string = max(probabilities, key=lambda kv: probabilities[kv])[::-1]
    # x_string = eigen_result.best_measurement['bitstring'][::-1]
    x = np.fromiter(x_string, dtype=int)

    print(f"Quantum solution: {x_string}, objective: {qp.objective.evaluate(x)}\
          \nQuantum measurements: {eigen_result.cost_function_evals}, time: {eigen_result.optimizer_time}")

    return eigen_result, trajectory, x_string, (operator, offset)


def maxcut_quantum_problem(weight_matrix, optimizer=SLSQP(maxiter=300), circuit='', initial_point=None, reps=1, backend=None):
    n = weight_matrix.shape[0]

    qp = QuadraticProgram("Max-Cut")
    qp.binary_var_list(n, "x")

    qubo_matrix = -weight_matrix
    qubo_vector = sum(row for row in weight_matrix)

    qp.maximize(quadratic=qubo_matrix, linear=qubo_vector)

    return QuantumProblem(qp).solve(optimizer=optimizer, circuit=circuit, initial_point=initial_point, p=reps, backend=backend)

