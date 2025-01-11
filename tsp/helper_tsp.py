from sys import path
path.append('..')
from energy_landscape import *
from problem_quantum import *

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from itertools import permutations
import warnings
warnings.filterwarnings('ignore')

from qiskit_optimization.problems import QuadraticProgram
from qiskit_optimization.converters import QuadraticProgramToQubo
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit_optimization.applications import Tsp
from qiskit_algorithms import SamplingVQE
from qiskit_algorithms.optimizers import SPSA, ADAM, COBYLA, SLSQP
from qiskit.circuit.library.n_local.qaoa_ansatz import QAOAAnsatz
from qiskit.circuit.library import TwoLocal
from qiskit.primitives import Sampler, StatevectorSampler, BackendSampler



def draw_tsp_graph(graph, route=None):
    n = graph.number_of_nodes()
    distances = nx.to_numpy_array(graph)
    pos = [graph.nodes[node]["pos"] for node in graph.nodes]
    # pos = nx.spring_layout(graph)

    if route:
        di_graph = nx.DiGraph()
        di_graph.add_nodes_from(graph)

        for i in range(n):
            j = (i + 1) % n
            di_graph.add_edge(route[i], route[j], 
                              weight=distances[route[i]][route[j]])

        graph = di_graph
    
    nx.draw_networkx(graph, pos, node_size=600)
    edge_labels = nx.get_edge_attributes(graph, "weight")
    nx.draw_networkx_edge_labels(graph, pos, edge_labels)
    plt.show()

def tsp_cost_function(bitstring, dist_matrix):
    distance = 0
    prev = 0
    for node in bitstring:
        distance += dist_matrix[node, prev]
        prev = node
    distance += dist_matrix[0, prev]

    return distance

@time_func
def tsp_brute(dist_matrix):
    n = len(dist_matrix)
    possible_routes = list(permutations(range(1, n)))
    best_distance = 1e10

    for route in possible_routes:
        distance = tsp_cost_function(route, dist_matrix)

        if distance < best_distance:
            best_route = (0,*route)
            best_distance = distance
            # print(f"order = {best_route} Distance = {distance}")

    return best_distance, best_route

def interpret(x):
    n = int(np.sqrt(len(x)))
    route = []
    for p in range(n):
        p_step = []
        for i in range(n):
            if x[i*n + p]:
                p_step.append(i)
        if len(p_step) == 1:
            route.extend(p_step)
        else:
            route.append(p_step)
    return route

def tsp_quantum(graph, optimizer=COBYLA(maxiter=300), circuit='', initial_point=None, reps=1, backend=None):
    n = graph.number_of_nodes()
    distances = nx.to_numpy_array(graph)
    sampler = BackendSampler(backend=backend) if backend else Sampler()

    qp = QuadraticProgram("Traveling Salesman")
    qp.binary_var_list([f"{i}{j}" for i in range(n) for j in range(n)], "x")
    
    qubo_matrix = np.zeros((n**2,n**2))
    for i in range(n):
        for j in range(n):
            for p in range(n):
                qubo_matrix[ n*i+p, n*j + (p+1)%n ] = distances[i,j]

    for i in range(n):
        qp.linear_constraint(linear={f"x{i}{k}":1 for k in range(n)}, sense="==", rhs=1)
    for k in range(n):
        qp.linear_constraint(linear={f"x{i}{k}":1 for i in range(n)}, sense="==", rhs=1)

    qp.minimize(quadratic=qubo_matrix)

    # Convert constrained quadratic problem to QUBO
    qubo = QuadraticProgramToQubo().convert(qp)

    # Map to the Ising problem
    operator, offset = qubo.to_ising()

    trajectory={'beta_0':[], 'gamma_0':[], 'energy':[]}
    def callback(eval_count, params, value, std_dev):
        trajectory['beta_0'].append(params[1])
        trajectory['gamma_0'].append(params[0])
        trajectory['energy'].append(value)

    # Choose the quantum circuit
    ry_ansatz = TwoLocal(operator.num_qubits, "ry", "cz", entanglement="linear", reps=reps)
    qaoa_ansatz = QAOAAnsatz(operator, reps=reps).decompose()
    ansatz = qaoa_ansatz if circuit.lower()=='qaoa' else ry_ansatz

    vqe = SamplingVQE(sampler, ansatz, optimizer, initial_point=initial_point, callback=callback)
    eigen_result = vqe.compute_minimum_eigenvalue(operator)

    #TODO: Check if this is correct
    # x = [int(i) for i in eigen_result.best_measurement['bitstring'][::-1]]
    # print("time:", eigen_result.optimizer_time)
    # print("qubits:", x)
    # print("solution:", interpret(x))
    # print("solution objective:", qubo.objective.evaluate(x))
    
    # Compute the most likely output
    probabilities = eigen_result.eigenstate.binary_probabilities()
    x_string = max(probabilities, key=lambda kv: probabilities[kv])[::-1]
    x = np.fromiter(x_string, dtype=int)
    route = interpret(x)

    print(f"Quantum solution: {route}, objective: {qubo.objective.evaluate(x)} time: {eigen_result.optimizer_time}")

    return eigen_result, trajectory, x, (operator, offset)


def tsp_quantum_problem(graph, optimizer=SLSQP(maxiter=300), circuit='', initial_point=None, reps=1, backend=None):
    n = graph.number_of_nodes()
    distances = nx.to_numpy_array(graph)

    qp = QuadraticProgram("Traveling Salesman")
    qp.binary_var_list([f"{i}{j}" for i in range(n) for j in range(n)], "x")
    
    qubo_matrix = np.zeros((n**2,n**2))
    for i in range(n):
        for j in range(n):
            for p in range(n):
                qubo_matrix[ n*i+p, n*j + (p+1)%n ] = distances[i,j]

    for i in range(n):
        qp.linear_constraint(linear={f"x{i}{k}":1 for k in range(n)}, sense="==", rhs=1)
    for k in range(n):
        qp.linear_constraint(linear={f"x{i}{k}":1 for i in range(n)}, sense="==", rhs=1)

    qp.minimize(quadratic=qubo_matrix)

    # Convert constrained quadratic problem to QUBO
    qubo = QuadraticProgramToQubo().convert(qp)

    return QuantumProblem(qubo).solve(optimizer=optimizer, circuit=circuit, initial_point=initial_point, reps=reps, backend=backend)

