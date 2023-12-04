import numpy as np
from qulacs.gate import RX, RY, RZ, PauliRotation, CNOT, CZ
from qiskit.quantum_info.operators.operator import Operator
from qiskit import QuantumCircuit as QiskitCircuit
from pytket import Circuit as TketCircuit
from pytket.circuit import Unitary1qBox, Unitary2qBox

def qulacs_to_qiskit(circuit):
    """
    Convert a Qulacs QuantumCircuit to a Qiskit QuantumCircuit.

    Parameters:
    circuit (QulacsCircuit): The Qulacs QuantumCircuit to convert.

    Returns:
    QiskitCircuit: The converted Qiskit QuantumCircuit.

    Raises:
    ValueError: If an invalid gate type is encountered.
    """
    circ_qiskit = QiskitCircuit(circuit.get_qubit_count())
    gate_groups = ['DenseMatrix', 'Pauli-rotation', 'X-rotation', 'Y-rotation', 'Z-rotation']
    for i in range(circuit.get_gate_count()):
        gate = circuit.get_gate(i)
        gate_type = gate.get_name()
        control = gate.get_control_index_list()
        target = gate.get_target_index_list()
        if gate_type == 'CNOT':
            circ_qiskit.cx(control[0], target[0])
        elif gate_type == 'CZ':
            circ_qiskit.cz(control[0], target[0])
        elif gate_type == 'X':
            circ_qiskit.x(target[0])
        elif gate_type == 'Y':
            circ_qiskit.y(target[0])
        elif gate_type == 'Z':
            circ_qiskit.z(target[0])
        elif gate_type == 'H':
            circ_qiskit.h(target[0])
        elif gate_type == 'S':
            circ_qiskit.s(target[0])
        elif gate_type == 'T':
            circ_qiskit.t(target[0])
        elif gate_type in gate_groups:
            unitary = gate.get_matrix()
            circ_qiskit.append(Operator(unitary), target)
        else:
            raise ValueError(f'Invalid gate type: {gate_type}')
    return circ_qiskit

def qulacs_to_tket(circuit_qulacs):
    """
    Convert a Qulacs QuantumCircuit to a Tket Circuit.

    Parameters:
    circuit_qulacs (QulacsCircuit): The Qulacs QuantumCircuit to convert.

    Returns:
    TketCircuit: The converted Tket Circuit.

    Raises:
    ValueError: If an invalid gate type is encountered.
    """
    nqubits = circuit_qulacs.get_qubit_count()
    circ_tket = TketCircuit(nqubits)
    for i in range(circuit_qulacs.get_gate_count()):
        gate = circuit_qulacs.get_gate(i)
        gate_type = gate.get_name()
        control = [(nqubits - 1) - i for i in gate.get_control_index_list()]
        control.reverse()
        target = [(nqubits - 1) - i for i in gate.get_target_index_list()]
        target.reverse()

        if gate_type == 'CNOT':
            circ_tket.CX(control[0], target[0])
        elif gate_type == 'CZ':
            circ_tket.CZ(control[0], target[0])
        elif gate_type == 'X':
            circ_tket.X(target[0])
        elif gate_type == 'Y':
            circ_tket.Y(target[0])
        elif gate_type == 'Z':
            circ_tket.Z(target[0])
        elif gate_type == 'H':
            circ_tket.H(target[0])
        elif gate_type == 'S':
            circ_tket.S(target[0])
        elif gate_type == 'T':
            circ_tket.T(target[0])
        elif gate_type in ['X-rotation', 'Y-rotation', 'Z-rotation']:
            unitary = gate.get_matrix()
            circ_tket.add_gate(Unitary1qBox(unitary), target)
        elif gate_type in ['DenseMatrix', 'Pauli-rotation']:
            unitary = gate.get_matrix()
            circ_tket.add_gate(Unitary2qBox(unitary), target)
        else:
            raise ValueError(f'Invalid gate type: {gate_type}')
    return circ_tket

def cirq_gates_to_qulacs(cirq_gates, dem_index):
    """
    Convert a list of Cirq gates to a list of Qulacs gates.

    Parameters:
    cirq_gates (List[cirq.Gate]): The list of Cirq gates to convert.
    dem_index (List[int]): The list of index.

    Returns:
    List[QuantumGateBase]: The converted list of Qulacs gates.

    Raises:
    ValueError: If an invalid gate type is encountered.
    """
    qulacs_gates = []
    one_qubit_gate_mapping = {
        'PhasedXPowGate': lambda gate, index: [RZ(index, gate.phase_exponent * np.pi), RX(index, -1 * gate.exponent * np.pi), RZ(index, -1 * gate.phase_exponent * np.pi)],
        'Rx': lambda gate, index: [RX(index, -1 * gate.exponent * np.pi)],
        'Ry': lambda gate, index: [RY(index, -1 * gate.exponent * np.pi)],
        'Rz': lambda gate, index: [RZ(index, -1 * gate.exponent * np.pi)]
    }
    two_qubit_gate_mapping = {
        'CXPowGate': lambda gate, index1, index2: [CNOT(index1, index2)],
        'CZPowGate': lambda gate, index1, index2: [CZ(index1, index2)],
        'XXPowGate': lambda gate, index1, index2: [PauliRotation([index1, index2], [1, 1], -1 * gate.exponent * np.pi)],
        'YYPowGate': lambda gate, index1, index2: [PauliRotation([index1, index2], [2, 2], -1 * gate.exponent * np.pi)],
        'ZZPowGate': lambda gate, index1, index2: [PauliRotation([index1, index2], [3, 3], -1 * gate.exponent * np.pi)]
    }
    for i in range(len(cirq_gates)):
        gate = cirq_gates[i].gate
        gate_name = type(gate).__name__
        index_list = [qubit.x for qubit in cirq_gates[i].qubits]
        index = dem_index[len(dem_index) - dem_index.index(index_list[0]) - 1]

        if len(index_list) == 1:
            if gate_name in one_qubit_gate_mapping:
                qulacs_gates.extend(one_qubit_gate_mapping[gate_name](gate, index))
            else:
                phase_z = gate.exponent * np.pi
                qulacs_gates.append(RZ(index, -1 * phase_z))
        elif len(index_list) == 2:
            index_2 = dem_index[len(dem_index) - dem_index.index(index_list[1]) - 1]
            if gate_name in two_qubit_gate_mapping:
                qulacs_gates.extend(two_qubit_gate_mapping[gate_name](gate, index, index_2))
            else:
                raise ValueError(f'Invalid gate type: {gate_name}')
        else:
            raise ValueError(f'Invalid gate type: {gate_name}')
    return qulacs_gates