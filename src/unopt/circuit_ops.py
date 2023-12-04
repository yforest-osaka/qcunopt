import numpy as np
from qulacs import QuantumCircuit
from qulacs.gate import SWAP, DenseMatrix


def swap_idx(gate):
    """
    Swap the target indices of a given gate.

    Args:
        gate (Gate): The gate whose target indices should be swapped.

    Returns:
        DenseMatrix: The result of swapping the target indices.
    """
    tg = list(reversed(gate.get_target_index_list()))
    swap = SWAP(0, 1).get_matrix()
    return DenseMatrix(tg, swap @ gate.get_matrix() @ swap)


def ctrl_gates_to_dem(circuit):
    """
    Convert controlled gates to DenseMatrix gates in a given quantum circuit.

    Args:
        circuit (QuantumCircuit): The quantum circuit to be modified.

    Returns:
        QuantumCircuit: The modified quantum circuit.
    """
    CZ_MATRIX = np.array(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, -1]])
    CNOT_MATRIX = np.array(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
    GATE_MATRIX_MAP = {
        'CZ': CZ_MATRIX,
        'CNOT': CNOT_MATRIX,
    }
    n = circuit.get_qubit_count()
    Circuit_out = QuantumCircuit(n)
    for g in range(circuit.get_gate_count()):
        gate = circuit.get_gate(g)
        gate_name = gate.get_name()
        if gate_name in GATE_MATRIX_MAP:  # CZ or CNOT
            ctrl, tg = gate.get_control_index_list()[0], gate.get_target_index_list()[0]
            Circuit_out.add_dense_matrix_gate([tg, ctrl], GATE_MATRIX_MAP[gate_name])
        else:
            Circuit_out.add_gate(gate)
    return Circuit_out
