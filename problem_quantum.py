import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import time
import warnings
warnings.filterwarnings('ignore')

from qiskit_optimization import QuadraticProgram
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit_optimization.applications import Maxcut
from qiskit_algorithms import QAOA, SamplingVQE, NumPyMinimumEigensolver
from qiskit_algorithms.optimizers import SPSA, ADAM, COBYLA, SLSQP
from qiskit.circuit.library.n_local.qaoa_ansatz import QAOAAnsatz
from qiskit.circuit.library import TwoLocal
from qiskit.primitives import Sampler, StatevectorSampler, BackendSampler, BackendSamplerV2


class QuantumProblem():
    # Constants
    default_color = '#1f78b4'
    seed = 123

    def __init__(self, qubo_program:QuadraticProgram):
        self.qubo_program = qubo_program

    def vqe_callback(self, trajectory, offset):
        def callback(eval_count, params, value, std_dev):
            trajectory['beta_0'].append(params[1])
            trajectory['gamma_0'].append(params[0])
            trajectory['energy'].append(-value -offset)
        return callback
    
    def solve(self, optimizer=SLSQP(maxiter=300), circuit='', initial_point=None, reps=1, backend=None):
        sampler = BackendSampler(backend=backend) if backend else Sampler()

        # Map to the Ising problem
        operator, offset = self.qubo_program.to_ising()

        # VQE callback function to save the trajectory of the optimization parameters
        trajectory={'beta_0':[], 'gamma_0':[], 'energy':[]}

        # Choose the quantum circuit
        ry_ansatz = TwoLocal(operator.num_qubits, "ry", "cz", entanglement="linear", reps=reps)
        qaoa_ansatz = QAOAAnsatz(operator, reps=reps).decompose()
        ansatz = qaoa_ansatz if circuit.lower()=='qaoa' else ry_ansatz

        # Choose initial points: [-.7, 4.4], [0.2, -0.2], [0.3, 6.1] for maxcut
        vqe = SamplingVQE(sampler, ansatz, optimizer, initial_point=initial_point, callback=self.vqe_callback(trajectory, offset))
        eigen_result = vqe.compute_minimum_eigenvalue(operator)

        # Compute the most likely output
        probabilities = eigen_result.eigenstate.binary_probabilities()
        x_string = max(probabilities, key=lambda kv: probabilities[kv])[::-1]
        x = np.fromiter(x_string, dtype=int)

        print(f"Quantum solution: {x_string}, objective: {self.qubo_program.objective.evaluate(x)} time: {eigen_result.optimizer_time}")

        return eigen_result, trajectory, x_string, (operator, offset)



