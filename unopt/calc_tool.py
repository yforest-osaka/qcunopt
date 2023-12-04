import cirq
import numpy as np
from qiskit.quantum_info import Statevector
from qulacs import QuantumState

def qulacs_func(circuit):
    """
    Evolve |00...0> state by a Qulacs circuit.

    Args:
        circuit (QuantumCircuit): The Qulacs quantum circuit to be modified.

    Returns:
        np.ndarray: The evolved state vector.
    """
    n = circuit.get_qubit_count()
    state = QuantumState(n)
    state.set_zero_state()
    circuit.update_quantum_state(state)
    return state.get_vector()

def cirq_func(circuit):
    """
    Evolve |00...0> state by a Cirq circuit.

    Args:
        circuit (cirq.Circuit): The Cirq quantum circuit to be modified.

    Returns:
        np.ndarray: The evolved state vector.
    """
    return cirq.final_state_vector(program=circuit, initial_state=0)

def qiskit_func(circuit):
    """
    Evolve |00...0> state by a Qiskit circuit.

    Args:
        circuit (qiskit.QuantumCircuit): The Qiskit quantum circuit to be modified.

    Returns:
        np.ndarray: The evolved state vector.
    """
    return Statevector.from_int(0, 2 ** circuit.num_qubits).evolve(circuit).data

def tket_func(circuit):
    """
    Evolve |00...0> state by a Tket circuit.

    Args:
        circuit (pytket.circuit.Circuit): The Tket quantum circuit to be modified.

    Returns:
        np.ndarray: The evolved state vector.
    """
    return circuit.get_statevector()

def operation_to_zero_state(Circuit):
    """
    Evolve |00...0> state by circuit (qulacs, cirq, qiskit, tket).

    Args:
        Circuit (QuantumCircuit): The quantum circuit to be modified.

    Returns:
        np.ndarray: The evolved state vector.
    """

    CIRCUIT_FUNC_MAP = {
        'qulacs': qulacs_func,
        'cirq': cirq_func,
        'qiskit': qiskit_func,
        'tket': tket_func
    }

    qc_type = str(type(Circuit))
    for circuit_type, func in CIRCUIT_FUNC_MAP.items():
        if circuit_type in qc_type:
            return func(Circuit)
    print('Circuit is invalid.')
    print(qc_type)
    print('-------------------')

def fidelity_check(Vector_1, Vector_2, phase = 'Calculation'):
    """
    Calculation of square inner product.

    Args:
        Vector_1 (np.ndarray): First state vector.
        Vector_2 (np.ndarray): Second state vector.
        phase (str, optional): Phase of the operation. Defaults to 'Calculation'.

    Returns:
        float: The fidelity of the two vectors.
    """
    fidelity = abs(np.dot(Vector_1, np.conjugate(Vector_2), out = None))**2
    if fidelity < 0.95:
        print(f'Fidelity is incorrect at phase {phase}!')
        print('#Fidelity_check: ',fidelity)
        print('-----------------------------')
    return fidelity