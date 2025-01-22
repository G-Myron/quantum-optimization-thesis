from typing import Tuple
import numpy as np
from pandas import DataFrame
import plotly.graph_objects as go
from qiskit.primitives import StatevectorEstimator, Sampler
from qiskit.quantum_info import SparsePauliOp


def find_energy(beta_gamma_pairs, weights_array, circuit, cost_function, num_qubits, ising):
    # Create dictionary with precomputed costs for all bitstrings
    costs = {}
    for i in range(2**num_qubits):
        bitstr = f'{i:0{num_qubits}b}'
        x = [int(bit) for bit in bitstr]
        costs[bitstr] = cost_function(x, weights_array)

    data_points = []
    for beta,gamma in beta_gamma_pairs:
        copy_circuit = circuit.copy()
        copy_circuit.remove_final_measurements()
        
        (operator, offset) = ising
        estimator_value = StatevectorEstimator().run([(copy_circuit, operator, [gamma, beta])]).result()[0].data.evs
        energy = -estimator_value.real - offset

        data_points.append({'beta': beta, 'gamma': gamma, 'energy': energy})

    return data_points



def plot_qaoa_energy_landscape(weights_array, circuit, cost_function, num_qubits:int, ising:Tuple[SparsePauliOp, float], *, show=True) -> dict:
    beta_gamma_pairs = [(b,g) for b in np.linspace( -np.pi/2, np.pi/2, 30) 
                        for g in np.linspace(0, np.pi/2, 30)]
    data_points = find_energy(beta_gamma_pairs, weights_array, circuit, cost_function, num_qubits, ising)

    # Create and display surface plot from data_points
    df = DataFrame(data_points)
    df = df.pivot(index='beta', columns='gamma', values='energy')
    matrix = df.to_numpy().T

    fig = go.Figure(data = go.Surface(x=df.index, y=df.columns, z=matrix))

    fig.update_layout(
        title='QAOA Energy Landscape',
        scene=dict(xaxis_title='β (beta)', yaxis_title='γ (gamma)', zaxis_title='Energy'),
    )

    if show: fig.show()

    return fig


def plot_qaoa_trajectory(trajectory, weights_array, circuit, cost_function, num_qubits, ising:Tuple[SparsePauliOp, float]):
    beta_0 = trajectory['beta_0']
    gamma_0 = trajectory['gamma_0']
    energy = [e.real for e in trajectory['energy']]

    # data_points = find_energy(zip(beta_0, gamma_0), weights_array, circuit, cost_function, num_qubits, operator, offset)
    # energy = [d['energy'] for d in data_points]

    # Create the figure
    fig = plot_qaoa_energy_landscape(weights_array, circuit, cost_function, num_qubits, ising, show=False)

    fig.add_trace(
        go.Scatter3d(
            x=beta_0, y=gamma_0, z=energy,
            mode = 'lines', line = {'color': energy, 'colorscale': 'Blues', 'width': 7},
            name = 'Optimization Trajectory'
        )
    )

    fig.update_layout(
        title='QAOA Energy Landscape and Optimization Path',
        scene=dict(xaxis_title='β (beta)', yaxis_title='γ (gamma)', zaxis_title='Energy'),
    )

    fig.show()

    return 

