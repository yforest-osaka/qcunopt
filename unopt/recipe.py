import numpy as np
from qulacs import QuantumCircuit
from qulacs.gate import DenseMatrix, RandomUnitary, SWAP
from qulacs.circuit import QuantumCircuitOptimizer as QCO
from unopt.converter import *
from unopt.generator import two_qubit_pauli_gates_generator
from unopt.decomp import(
    two_qubit_gate_decomposition,
    three_qubit_gate_decomposition,
)
from unopt.circuit_ops import swap_idx
import networkx as nx

def find_unopt_pair(circuit, position):
    """
    Given a quantum circuit, this function aims to find pairs of two-qubit DeM gates. 
    The search is conducted both forwards and backwards from a given starting position.
    
    Args:
        circuit (QuantumCircuit): The quantum circuit in which the search will be performed.
        position (int): The position in the circuit from where the search will start.
        
    Returns:
        list: A list containing the indices of the gate pairs.
    """
    forward_indices, backward_indices = [], []
    total_gates = circuit.get_gate_count()
    start_gate_targets = circuit.get_gate(position).get_target_index_list()
    upper_gate_found, lower_gate_found = False, False
    involved_qubits = []

    for i in range(position + 1, total_gates):
        if upper_gate_found and lower_gate_found:
            break
        current_gate_targets = circuit.get_gate(i).get_target_index_list()
        common_targets = list(set(start_gate_targets) & set(current_gate_targets))

        if len(common_targets) >= 2:
            break
        elif len(common_targets) == 1:
            if common_targets[0] == start_gate_targets[0] and not upper_gate_found:
                upper_gate_found = True
                involved_qubits.append(current_gate_targets)
                backward_indices.append(i)
            elif common_targets[0] == start_gate_targets[1] and not lower_gate_found:
                lower_gate_found = True
                involved_qubits.append(current_gate_targets)
                backward_indices.append(i)
            else:
                involved_qubits.append(current_gate_targets)
        else:
            involved_qubits.append(current_gate_targets)

    if involved_qubits:
        involved_qubits.pop()

    if len(backward_indices) == 2:
        second_gate_targets = circuit.get_gate(backward_indices[1]).get_target_index_list()
        target_index = list(set(start_gate_targets) & set(second_gate_targets))[0]
        for indices in involved_qubits:
            if target_index in indices:
                backward_indices = [backward_indices[0]]

        gate_graph = nx.Graph()
        non_common_targets = [i for i in start_gate_targets + second_gate_targets if (i not in start_gate_targets) or (i not in second_gate_targets)]
        for indices in involved_qubits:
            gate_graph.add_edge(indices[0], indices[1])
        connected_component = nx.node_connected_component(gate_graph, non_common_targets[0])
        if non_common_targets[1] in connected_component:
            backward_indices = [backward_indices[0]]

    upper_gate_found, lower_gate_found = False, False
    involved_qubits = []
    start_gate_targets = circuit.get_gate(position).get_target_index_list()

    for i in reversed(range(position)):
        if upper_gate_found and lower_gate_found:
            break
        current_gate_targets = circuit.get_gate(i).get_target_index_list()
        common_targets = list(set(current_gate_targets) & set(start_gate_targets))

        if len(common_targets) >= 2:
            break
        elif len(common_targets) == 1:
            if common_targets[0] == start_gate_targets[0] and not upper_gate_found:
                upper_gate_found = True
                involved_qubits.append(current_gate_targets)
                forward_indices.append(i)
            elif common_targets[0] == start_gate_targets[1] and not lower_gate_found:
                lower_gate_found = True
                involved_qubits.append(current_gate_targets)
                forward_indices.append(i)
            else:
                involved_qubits.append(current_gate_targets)
        else:
            involved_qubits.append(current_gate_targets)

    if involved_qubits:
        involved_qubits.pop()

    if len(forward_indices) == 2:
        second_gate_targets = circuit.get_gate(forward_indices[1]).get_target_index_list()
        target_index = list(set(start_gate_targets) & set(second_gate_targets))[0]
        for indices in involved_qubits:
            if target_index in indices:
                forward_indices = [forward_indices[0]]

        gate_graph = nx.Graph()
        non_common_targets = [i for i in start_gate_targets + second_gate_targets if (i not in start_gate_targets) or (i not in second_gate_targets)]
        for indices in involved_qubits:
            gate_graph.add_edge(indices[0], indices[1])
        connected_component = nx.node_connected_component(gate_graph, non_common_targets[0])
        if non_common_targets[1] in connected_component:
            forward_indices = [forward_indices[0]]

    return forward_indices + backward_indices

def generate_unitary_gate(target_indices, inserted_gate_type, pattern):
    """
    Generate a unitary gate based on the inserted_gate type, target_indices and pattern.

    Args:
        target_indices (list): The target indices for the gate.
        inserted_gate_type (str): Type of the gate to be inserted. It can be 'random' or 'PauliZZ'.
        pattern (str): The pattern of the unoptimized gate. It can be 'A', 'B', or 'C'.

    Returns:
        Gate: The generated unitary gate.
    """
    if pattern != 'B':
        if inserted_gate_type == 'random':
            return RandomUnitary([target_indices[1], target_indices[2]])
        elif inserted_gate_type == 'pauli':
            print('PauliRandom!')
            return two_qubit_pauli_gates_generator([target_indices[1], target_indices[2]], np.pi / 4, pauli_type = 'random')
    else:
        if inserted_gate_type == 'random':
            return RandomUnitary([target_indices[0], target_indices[1]])
        elif inserted_gate_type == 'pauli':
            print('PauliRandom!')
            return two_qubit_pauli_gates_generator([target_indices[0], target_indices[1]], np.pi / 4, pauli_type = 'random')

def generate_U_dagger_tilde_gate(target_indices, gate_A, U_dagger, pattern):
    """
    Generate the U_dagger_tilde_gate.

    Args:
        target_indices (list): The target indices for the gate.
        gate_A (Gate): The gate A in the circuit.
        U_dagger (np.ndarray): The conjugate transpose of the unitary gate U.
        pattern (str): The pattern of the unoptimized gate. It can be 'A', 'B', or 'C'.

    Returns:
        DenseMatrix: The generated U_dagger_tilde gate.
    """
    if pattern == 'A':
        A = np.kron(np.eye(2), gate_A.get_matrix())
    elif pattern == 'B':
        A = np.kron(gate_A.get_matrix(), np.eye(2))
    else:
        B = np.kron(np.eye(2), gate_A.get_matrix())
        swap = np.kron(SWAP(0,1).get_matrix(), np.eye(2))
        A = swap @ B @ swap

    A_dagger =  np.conjugate(A).T
    if pattern != 'B':
        U_dagger_tilde = A_dagger @ np.kron(U_dagger, np.eye(2)) @ A
    else:
        U_dagger_tilde = A_dagger @ np.kron(np.eye(2), U_dagger) @ A
    return DenseMatrix(target_indices, U_dagger_tilde)

def elementary_recipe(qulacs_circuit, first_index, second_index, inserted_gate_type):
    """
    This function generates an elementary recipe by splitting the input circuit into three parts,
    adding an arbitrary gate, and then rejoining the parts.
    
    Args:
        qulacs_circuit (QuantumCircuit): The input quantum circuit.
        first_index (int): The first index to split the circuit.
        second_index (int): The second index to split the circuit.
        inserted_gate_type (str): The type of gate to be inserted('random' or 'PauliZZ').
        
    Returns:
        QuantumCircuit: The optimized quantum circuit (all).
        QuantumCircuit: The optimized part of the circuit.
        int: The number of gates in the first part of the unoptimized circuit.
        list: The target indices of the first gate.
        list: The target indices of the second gate.
    """
    nqubits = qulacs_circuit.get_qubit_count()
    circuit_front = QuantumCircuit(nqubits)
    circuit_middle = QuantumCircuit(nqubits)
    circuit_back = QuantumCircuit(nqubits)
    
    # Split into three circuits
    for j in range (first_index):
        circuit_front.add_gate(qulacs_circuit.get_gate(j))
    
    gate_graph = nx.Graph()
    target_a = qulacs_circuit.get_gate(first_index).get_target_index_list()
    gate_graph.add_edge(target_a[0],target_a[1])
    for j in range (first_index + 1, second_index):
        current_targets = qulacs_circuit.get_gate(j).get_target_index_list()
        gate_graph.add_edge(current_targets[0],current_targets[1])
        connected_component = nx.node_connected_component(gate_graph, current_targets[0])
        if target_a[0] in connected_component or target_a[1] in connected_component: # If overlapping with target_a, add to the middle part
            circuit_middle.add_gate(qulacs_circuit.get_gate(j))
        else: # Otherwise, add to the front part
            circuit_front.add_gate(qulacs_circuit.get_gate(j))
            gate_graph.remove_edge(current_targets[0],current_targets[1])

    # apply elementary recipe
    for j in range (second_index + 1 ,qulacs_circuit.get_gate_count()):
        circuit_back.add_gate(qulacs_circuit.get_gate(j))
    gate_1 = qulacs_circuit.get_gate(first_index)
    gate_2 = qulacs_circuit.get_gate(second_index)
    gate_1_targets = gate_1.get_target_index_list()
    gate_2_targets = gate_2.get_target_index_list()
    target_indices = sorted(set(gate_1_targets + gate_2_targets)) # Get three qubit indices

    # Classification of unoptimized patterns (A,B,C)
    calc = [i for i in target_indices if i not in gate_1_targets]
    pattern = 'None'
    if calc[0] == target_indices[2]:
        pattern = 'A'
    elif calc[0] == target_indices[0]:
        pattern = 'B'
    # Change indices if DenseMatrix is reversed
    if gate_1_targets[1] - gate_1_targets[0] < 0:
        gate_1 = swap_idx(gate_1)

    # Generate the arbitrary gate for insertion
    unitary_gate = generate_unitary_gate(target_indices, inserted_gate_type, pattern)
    U_dagger = np.conjugate(unitary_gate.get_matrix()).T

    U_dagger_tilde_gate = generate_U_dagger_tilde_gate(target_indices, gate_1, U_dagger, pattern)
    
    circuit_main = QuantumCircuit(nqubits)

    decomposition_result = three_qubit_gate_decomposition(U_dagger_tilde_gate, nqubits)
    for gate in decomposition_result:
        circuit_main.add_gate(gate)
    decomposition_result_2 = two_qubit_gate_decomposition(gate_1, nqubits)
    decomposition_result_3 = two_qubit_gate_decomposition(unitary_gate, nqubits)
    decomposition_result_4 = two_qubit_gate_decomposition(gate_2, nqubits)
    gates = decomposition_result_2 + decomposition_result_3 + decomposition_result_4
    for gate in gates:
        circuit_main.add_gate(gate)
    
    optimizer = QCO()
    max_block_size = 2
    optimizer.optimize(circuit_main,max_block_size)
    
    # Joint all circuits
    output_circuit = QuantumCircuit(nqubits)
    output_circuit.merge_circuit(circuit_front)
    output_circuit.merge_circuit(circuit_main)
    output_circuit.merge_circuit(circuit_middle)
    output_circuit.merge_circuit(circuit_back)
    gate_num_info = circuit_front.get_gate_count()

    return output_circuit, circuit_main, gate_num_info, gate_1_targets, gate_2_targets