from sys import path
path.append('..')
from energy_landscape import *
from problem_quantum import *

import warnings
warnings.filterwarnings('ignore')

from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.circuit.library import TwoLocal, QAOAAnsatz
from qiskit.primitives import Sampler, BackendSampler
from qiskit_algorithms import SamplingVQE
from qiskit_algorithms.optimizers import COBYLA, SPSA, SLSQP, ADAM
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.algorithms import OptimizationAlgorithm
from qiskit_optimization.converters import QuadraticProgramToQubo


def make_knapsack_cost_function(qubo_converter:QuadraticProgramToQubo=None):
    interpret = qubo_converter.interpret if qubo_converter else lambda x: x
    def knapsack_cost_function(bitstring, profits):
        x = interpret(bitstring)
        return sum([x[i]*profits[i] for i in range(len(x))])
    return knapsack_cost_function

@time_func
def knapsack_dynamic(capacity, weights, values):
    n = len(weights)

    # Setup a table with W+1 columns and n+1 rows
    matrix = [[0 for _ in range(capacity + 1)] for _ in range(n + 1)]

    # Fill the table in bottom up manner
    for i in range(n + 1):
        for w in range(capacity + 1):
            # First row and column are zero
            if i == 0 or w == 0:
                matrix[i][w] = 0
            # 
            elif weights[i - 1] <= w:
                matrix[i][w] = max(values[i - 1] + matrix[i - 1][w - weights[i - 1]], matrix[i - 1][w])
            else:
                matrix[i][w] = matrix[i - 1][w]

    # Find the selected items
    best_value = matrix[n][capacity]
    selected_items = []
    result = np.zeros(n)
    new_capacity = capacity
    for i in range(n-1, -1, -1):
        if matrix[i][new_capacity] < best_value:
            selected_items.append((i, weights[i], values[i]))
            result[i] = 1
            new_capacity -= weights[i]
            best_value -= values[i]

    return matrix[n][capacity], selected_items[::-1], result


def knapsack_quantum(profits, weights, max_weight, *,
                     optimizer=COBYLA(maxiter=300), circuit='', initial_point=None, p=1, backend=None):
    n = len(weights)
    sampler = BackendSampler(backend=backend) if backend else Sampler()

    # Define the Quadratic Program for the problem
    qp = QuadraticProgram("Knapsack")
    qp.binary_var_list(n, "x")

    qp.maximize(linear=profits)
    qp.linear_constraint(weights, "<=", max_weight)

    # Map to the Ising model
    qubo_converter = QuadraticProgramToQubo()
    qubo = qubo_converter.convert(qp)
    operator, offset = qubo.to_ising()

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
        ansatz = TwoLocal(operator.num_qubits, "ry", "cz", entanglement="full", reps=p)

    # Create and run the VQE
    vqe = SamplingVQE(sampler, ansatz, optimizer, initial_point=initial_point, callback=callback)
    eigen_result = vqe.compute_minimum_eigenvalue(operator)

    # Get the best solution from the statevector
    raw_samples = OptimizationAlgorithm._eigenvector_to_solutions(eigen_result.eigenstate, qubo)
    raw_samples.sort(key=lambda x: x.fval)
    x = qubo_converter.interpret(raw_samples[0].x)

    print(f"Quantum solution: {x}, value: {qp.objective.evaluate(x)}\
          \nQuantum measurements: {eigen_result.cost_function_evals}, time: {eigen_result.optimizer_time}")

    return eigen_result, trajectory, x, (operator, offset), qubo_converter


def knapsack_quantum_problem(profits, weights, max_weight, *,
                             initial_point=None, optimizer=COBYLA(), circuit='', reps=1, backend=None):
    n = len(weights)

    qp = QuadraticProgram("Knapsack")
    qp.binary_var_list(n, "x")

    qp.maximize(linear=profits)
    qp.linear_constraint(weights, "<=", max_weight)

    # Map to the Ising problem
    qubo_converter = QuadraticProgramToQubo()
    qubo = qubo_converter.convert(qp)

    result = QuantumProblem(qubo).solve(optimizer=optimizer, circuit=circuit, initial_point=initial_point, p=reps, backend=backend)

    return (*result, qubo_converter)

