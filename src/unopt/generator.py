import numpy as np
from qulacs import QuantumCircuit
from qulacs.gate import PauliRotation, RandomUnitary
import random

def two_qubit_pauli_gates_generator(list_of_place, angle, pauli_type):
    """
    Generate two-qubit Pauli gates.

    Parameters:
    list_of_place (list): List of places where the operation should be applied.
    angle (float): The angle of rotation.
    pauli_type (str): The type of Pauli operation. Can be 'XX', 'YY', 'ZZ', or 'RANDOM'.

    Returns:
    PauliRotation: The generated two-qubit Pauli gate.
    """
    pauli_type = pauli_type.upper()

    pauli_dict = {
        'XX': [1, 1],
        'YY': [2, 2],
        'ZZ': [3, 3],
        'RANDOM': [np.random.randint(1,4) for _ in range(2)]
    }

    pauli_strings = pauli_dict.get(pauli_type)
    
    if pauli_strings is None:
        raise ValueError("Pauli type is invalid.")
    
    rotation_operator = PauliRotation(list_of_place, pauli_strings, -1 * angle)
    return rotation_operator

def generate_random_qc(nqubits, depth):
    """
    Generate a random quantum circuit.

    Parameters:
    nqubits (int): The number of qubits.
    depth (int): The depth of the circuit.

    Returns:
    QuantumCircuit: The generated quantum circuit.
    """
    circuit = QuantumCircuit(nqubits)
    qubit_index = list(range(nqubits))
    for _ in range(depth):
        random.shuffle(qubit_index)
        for k in range(nqubits // 2):
            targets = [qubit_index[2*k], qubit_index[2*k+1]]
            circuit.add_gate(RandomUnitary(targets))
    return circuit