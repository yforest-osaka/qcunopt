import numpy as np
from qulacs.gate import *
import cirq
from unopt.converter import cirq_gates_to_qulacs

def kak_decomposition(gate, nqubits): 
    """
    This function decomposes the given 'gate' into a list of Cirq gates using the KAK decomposition method.

    Args:
        gate (qulacs_core.QuantumGateMatrix) : Quantum gate you want to decompose
        nqubits (int) : Number of qubits in the quantum circuit

    Returns:
        cirq_gates (list) : List of decomposed Cirq gates
    """
    
    # Prepare Cirq circuit
    circuit = cirq.LineQubit.range(nqubits) 

    # Get the indices of the target gate
    targets = gate.get_target_index_list() 

    cirq_gates = []
    
    # Perform KAK decomposition using Cirq
    kak_operations = cirq.kak_decomposition(gate.get_matrix()) 
    
    # Get the tuple of coefficients(X,Y,Z)
    coef = kak_operations.interaction_coefficients
    
    # Get the tuple of gates before and after KAK decomposition
    initial_operations = kak_operations.single_qubit_operations_before 
    final_operations = kak_operations.single_qubit_operations_after 

    # Add initial operations to the list of Cirq gates
    cirq_gates.append(cirq.MatrixGate(initial_operations[0])(circuit[targets[0]]))
    cirq_gates.append(cirq.MatrixGate(initial_operations[1])(circuit[targets[1]]))
    
    # Add middle operations (XX+YY+ZZ) to the list of Cirq gates
    cirq_gates.append(cirq.XXPowGate(exponent=-2 / np.pi * coef[0]).on(circuit[targets[0]],circuit[targets[1]]))
    cirq_gates.append(cirq.YYPowGate(exponent=-2 / np.pi * coef[1]).on(circuit[targets[0]],circuit[targets[1]]))
    cirq_gates.append(cirq.ZZPowGate(exponent=-2 / np.pi * coef[2]).on(circuit[targets[0]],circuit[targets[1]]))
    
    # Add final operations to the list of Cirq gates
    cirq_gates.append(cirq.MatrixGate(final_operations[0])(circuit[targets[0]]))
    cirq_gates.append(cirq.MatrixGate(final_operations[1])(circuit[targets[1]]))

    return cirq_gates

def euler_decomposition(cirq_gate, nqubits):
    """
    Decompose a given single qubit cirq gate into a list of gates 
    using euler decomposition.
    
    Args:
        cirq_gate (cirq.ops.gate_operation.GateOperation) : Single qubit cirq gate to decompose
        nqubits (int) : Number of qubits in the quantum circuit

    Returns:
        cirq_gates (list) : List of decomposed Cirq gates
    """
    cirq_gates = []
    
    # Create a cirq circuit and append the gate
    qc = cirq.Circuit()
    qc.append(cirq_gate)
    
    # Get the unitary matrix of the input gate
    unitary_matrix = qc.unitary()
    
    # Prepare Cirq circuit
    circuit = cirq.LineQubit.range(nqubits)

    # Get the target qubit
    target_qubit = int(cirq_gate.qubits[0])
    
    # Perform Euler decomposition
    decomposition_angles = cirq.deconstruct_single_qubit_matrix_into_angles(unitary_matrix)
    
    # Add decomposed gates to the list
    cirq_gates.append(cirq.rz(decomposition_angles[0])(circuit[target_qubit]))
    cirq_gates.append(cirq.ry(decomposition_angles[1])(circuit[target_qubit]))
    cirq_gates.append(cirq.rz(decomposition_angles[2])(circuit[target_qubit]))
    
    return cirq_gates

def two_qubit_gate_decomposition(qulacs_gate, nqubits):
    """
    Perform KAK decomposition of a given two qubit Qulacs gate. 
    
    Args:
        qulacs_gate ('qulacs_core.QuantumGateMatrix') : Two qubit Qulacs gate to decompose
        nqubits (int) : Number of qubits in the quantum circuit

    Returns:
        qulacs_gates (list) : List of decomposed Qulacs gates
    """
    
    # Perform KAK decomposition
    kak_operations = kak_decomposition(qulacs_gate, nqubits)
    
    # Get target indices
    targets = qulacs_gate.get_target_index_list()
    
    # Perform Euler decomposition for initial and final operations separately
    initial_op0 = euler_decomposition(kak_operations[0], nqubits)
    initial_op1 = euler_decomposition(kak_operations[1], nqubits)
    final_op0 = euler_decomposition(kak_operations[5], nqubits)
    final_op1 = euler_decomposition(kak_operations[6], nqubits)
    
    # Construct list of operations: initial operations -> KAK operations -> final operations
    decomposed_operations = initial_op0 + initial_op1 + kak_operations[2:5] + final_op0 + final_op1
    
    # Convert Cirq gates to Qulacs gates
    qulacs_gates = cirq_gates_to_qulacs(decomposed_operations, targets)
    
    return qulacs_gates

def three_qubit_gate_decomposition(qulacs_gate, nqubits):
    """
    Decomposes a given three qubit Qulacs gate into a list of decomposed gates

    Args:
        qulacs_gate (qulacs.QuantumGateBase) : Three qubit Qulacs gate to decompose
        nqubits (int) : Number of qubits in the quantum circuit

    Returns:
        qulacs_gates (list) : List of decomposed Qulacs gates
    """
    
    # Get the target indices of the gate
    targets = qulacs_gate.get_target_index_list()
    
    # Prepare the Cirq circuit
    circuit = cirq.LineQubit.range(nqubits)
    
    # Use Cirq's function to get the operations for a three-qubit gate
    cirq_operations = cirq.three_qubit_matrix_to_operations(circuit[targets[0]], circuit[targets[1]], circuit[targets[2]], qulacs_gate.get_matrix(), atol=1.0e-08)

    # Convert Cirq gates to Qulacs gates
    qulacs_gates = cirq_gates_to_qulacs(cirq_operations, targets)

    return qulacs_gates