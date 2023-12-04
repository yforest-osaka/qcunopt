from qulacs.circuit import QuantumCircuitOptimizer as QCO
import time
import pandas as pd
from tqdm import tqdm
from unopt.converter import qulacs_to_qiskit, qulacs_to_tket
from unopt.recipe import *
from unopt.circuit_ops import ctrl_gates_to_dem
from unopt.generator import generate_random_qc
from unopt.calc_tool import fidelity_check, operation_to_zero_state
import random
from qiskit.compiler import transpile
from qiskit.transpiler.passes import Optimize1qGatesDecomposition
from pytket import OpType
from pytket.passes import FullPeepholeOptimise, RebaseTket
from pytket.passes import auto_rebase_pass, DecomposeBoxes
from pytket.transform import Transform


def circ_conversion(circuit_initial, circuit_unopted, v_original):
    """
    This function converts a qulacs circuit to qiskit and tket circuits.
    It also checks the fidelity of the circuits.

    Parameters:
    circuit_initial: Initial qulacs circuit
    circuit_unopted: Unoptimized qulacs circuit
    v_original: Original vector to zero state

    Returns:
    circuit_qiskit: Converted qiskit circuit
    circuit_qiskit_with_unopt: Converted qiskit circuit with  unopt
    circuit_tket: Converted tket circuit
    circuit_tket_with_unopt: Converted tket circuit with unopt
    """

    # convert qulacs to qiskit
    circuit_qiskit = qulacs_to_qiskit(circuit_initial)
    fidelity_check(operation_to_zero_state(circuit_qiskit),
                   v_original, 'Qiskit conversion')
    circuit_qiskit_with_unopt = qulacs_to_qiskit(circuit_unopted)
    fidelity_check(operation_to_zero_state(
        circuit_qiskit_with_unopt), v_original, 'Qiskit conversion')

    # convert qulacs to tket
    circuit_tket = qulacs_to_tket(circuit_initial)
    if circuit_tket.n_qubits <= 11:
        fidelity_check(operation_to_zero_state(circuit_tket),
                       v_original, 'Tket conversion')

    circuit_tket_with_unopt = qulacs_to_tket(circuit_unopted)
    if circuit_tket_with_unopt.n_qubits <= 11:
        fidelity_check(operation_to_zero_state(
            circuit_tket_with_unopt), v_original, 'Tket conversion')

    assert circuit_qiskit.depth() == circuit_tket.depth()
    assert circuit_qiskit_with_unopt.depth() == circuit_tket_with_unopt.depth()

    return circuit_qiskit, circuit_qiskit_with_unopt, circuit_tket, circuit_tket_with_unopt


def qiskit_comp(circuit_qiskit, circuit_qiskit_with_unopt, v_original):
    """
    This function optimizes a qiskit circuit.
    It also creates a dataframe that contains information about the circuit before and after optimization.

    Parameters:
    circuit_qiskit: Initial qiskit circuit
    circuit_qiskit_with_unopt: Qiskit circuit with unopt
    v_original: Original vector to zero state

    Returns:
    df_result_qiskit: DataFrame containing information about the circuit before and after optimization
    """

    # Optimize circuits
    backend = None
    basis_gates = ['u3', 'cx']
    optimized_0 = transpile(circuit_qiskit, backend=backend,
                            seed_transpiler=11, optimization_level=0, basis_gates=basis_gates)
    optimized_0 = Optimize1qGatesDecomposition(basis=basis_gates)(
        optimized_0)  # this is the standard depth

    optimized_0_with_unopt = transpile(circuit_qiskit_with_unopt, backend=backend,
                                       seed_transpiler=11, optimization_level=0, basis_gates=basis_gates)
    optimized_3_with_unopt = transpile(circuit_qiskit_with_unopt, backend=backend,
                                       seed_transpiler=11, optimization_level=3, basis_gates=basis_gates)
    optimized_0_with_unopt = Optimize1qGatesDecomposition(
        basis=['u3'])(optimized_0_with_unopt)

    # Fidelity check
    for circuit in [optimized_0, optimized_0_with_unopt, optimized_3_with_unopt]:
        fidelity_check(operation_to_zero_state(circuit), v_original)

    # Create DataFrame
    df_result_qiskit = pd.DataFrame({
        'qiskit_u_bef_decomp': [circuit_qiskit.depth(), circuit_qiskit.size(), dict(circuit_qiskit.count_ops())],
        'qiskit_u': [optimized_0.depth(), optimized_0.size(), dict(optimized_0.count_ops())['cx']],
        'qiskit_v_bef_decomp': [circuit_qiskit_with_unopt.depth(), circuit_qiskit_with_unopt.size(), dict(circuit_qiskit_with_unopt.count_ops())],
        'qiskit_v': [optimized_0_with_unopt.depth(), optimized_0_with_unopt.size(), dict(optimized_0_with_unopt.count_ops())['cx']],
        'qiskit_v_compiled': [optimized_3_with_unopt.depth(), optimized_3_with_unopt.size(), dict(optimized_3_with_unopt.count_ops())['cx']]
    }, index=['Depth', 'Total gates', 'CNOT gates']).T

    return df_result_qiskit


def tket_comp(circuit_tket, circuit_tket_with_unopt, v_original):
    """
    This function optimizes a tket circuit.
    It also creates a dataframe that contains information about the circuit before and after optimization.

    Parameters:
    circuit_tket: Initial tket circuit
    circuit_tket_with_unopt: Tket circuit with unopt
    v_original: Original vector to zero state

    Returns:
    df_result_tket: DataFrame containing information about the circuit before and after optimization
    """

    # Initialize DataFrame
    df_result_tket = pd.DataFrame()

    # Optimize circuits
    for circuit, label in zip([circuit_tket, circuit_tket_with_unopt], ['tket_u', 'tket_v']):
        # Before optimization
        df_result_tket[f'{label}_bef_decomp'] = [
            circuit.depth(),
            circuit.n_gates,
            circuit.n_gates_of_type(OpType.CX)
        ]

        # Apply decompositions
        DecomposeBoxes().apply(circuit)
        RebaseTket().apply(circuit)
        merged_circuit = circuit.copy()
        Transform.ReduceSingles().apply(merged_circuit)
        auto_rebase_pass({OpType.CX, OpType.U3}).apply(merged_circuit)

        # Fidelity check
        if merged_circuit.n_qubits <= 11:
            fidelity_check(operation_to_zero_state(merged_circuit), v_original)

        # After RebaseTket optimization
        df_result_tket[f'{label}'] = [
            merged_circuit.depth(),
            merged_circuit.n_gates,
            merged_circuit.n_gates_of_type(OpType.CX)
        ]

    # Optimization for circuit with unopt
    FullPeepholeOptimise().apply(circuit_tket_with_unopt)
    auto_rebase_pass({OpType.CX, OpType.U3}).apply(circuit_tket_with_unopt)

    # Fidelity check
    if circuit_tket_with_unopt.n_qubits <= 11:
        fidelity_check(operation_to_zero_state(
            circuit_tket_with_unopt), v_original)

    # After FullPeepholeOptimise optimization
    df_result_tket['tket_v_compiled'] = [
        circuit_tket_with_unopt.depth(),
        circuit_tket_with_unopt.n_gates,
        circuit_tket_with_unopt.n_gates_of_type(OpType.CX)
    ]

    df_result_tket = df_result_tket.rename(
        index={0: 'Depth', 1: 'Total gates', 2: 'CNOT gates'}).T

    return df_result_tket


def random_pair_selection(circuit):
    """
    Select a random pair from the circuit.

    Args:
        circuit (object): The circuit from which to select pairs.

    Returns:
        list: A sorted list containing the random index and the selected pair.
    """
    while True:
        random_index = np.random.randint(0, circuit.get_gate_count()-1)
        candidates = find_unopt_pair(circuit, random_index)
        if candidates:
            pair = sorted([random_index, random.choice(candidates)])
            return pair


def concatenated_pair_selection(circuit, range_min, range_max, used_qubit_indices):
    """
    Select a concatenated pair from the circuit.

    Args:
        circuit (object): The circuit from which to select pairs.
        range_min (int): The minimum gate index of the previous ER.
        range_max (int): The maximum gate index of the previous ER.
        used_qubit_indices (list): List of qubit indices that have been used in the previous ER.

    Returns:
        list: A sorted list containing the index and the selected pair.
    """
    answers = []
    for index in range(range_min+1, range_max):
        candidates = find_unopt_pair(circuit, index)
        nums = [num for num in candidates if num <
                range_min or num > range_max]
        for num in nums:
            target_indices = circuit.get_gate(num).get_target_index_list()
            if target_indices[0] not in used_qubit_indices or target_indices[1] not in used_qubit_indices:
                answers.append(sorted([index, num]))
    if answers:
        pair = max(answers)
    else:
        pair = random_pair_selection(circuit)
    return pair


def unoptimization(circuit_input, iteration, pair_select_method, param1 = 'random'):
    """
    Perform unoptimization on the input circuit.

    Args:
        circuit_input: The input circuit to be unoptimized.
        iteration (int): The number of iterations to perform unoptimization.
        pair_select_method (str): The method to select pairs. Options are 'random' or 'concatenated'.
        param1 (str): The type of inserted gate. Options are 'random' or 'pauli'.

    Returns:
        tuple: A tuple containing the unoptimized circuit and the unoptimization level.
    """
    circuit_initial = circuit_input.copy()
    v_0 = operation_to_zero_state(circuit_initial)

    for i in tqdm(range(iteration), leave=False, desc='Unoptimization'):
        if pair_select_method == 'random':
            pair = random_pair_selection(circuit_input)
        elif pair_select_method == 'concatenated':
            if i == 0:
                pair = random_pair_selection(circuit_input)
            else:
                pair = concatenated_pair_selection(circuit_input, unopt_range_min, unopt_range_max, used)
        else:
            raise ValueError('Invalid pair selection method.')

        ops = elementary_recipe(
            circuit_input, first_index=pair[0], second_index=pair[1], inserted_gate_type=param1)

        second_position, generated_gate_counts = ops[2], ops[1].get_gate_count()
        unopt_range_min, unopt_range_max = second_position, second_position + generated_gate_counts - 1
        circuit_unopted = ctrl_gates_to_dem(ops[0])
        used = list(set(ops[3] + ops[4]))

        if fidelity_check(v_0, operation_to_zero_state(circuit_unopted)) < 0.95:
            break
        circuit_input = circuit_unopted.copy()

    c_original, c_unopt = circuit_initial.copy(), circuit_unopted.copy()
    QCO().optimize(c_original, 3)
    QCO().optimize(c_unopt, 3)
    unopt_level = c_unopt.calculate_depth() / c_original.calculate_depth()

    return circuit_unopted, unopt_level


def qc_benchmark(Number_of_qubits, Depth, Iteration, pair_select_method):
    """
    Compiler benchmark.

    Parameters:
    Number_of_qubits (int): Number of qubits.
    Depth (int): Depth of the quantum circuit.
    Iteration (int): Number of iterations.
    pair_select_method: Method to select pair.

    Returns:
    df_qiskit, df_tket, df_time, unopt_level: Benchmark results.
    """
    st_time = time.time()
    trial = 10
    flag = False
    # 1. Input_circuit
    for _ in range(trial):
        circuit_input = generate_random_qc(nqubits=Number_of_qubits, depth=Depth)        
        for p in range(circuit_input.get_gate_count()):
            if len(find_unopt_pair(circuit_input, p)) > 0:
                flag = True
                break
        else:
            continue
        break
    assert flag, f'Failed to generate an input circuit after {trial} trials. Please try again.'
    circuit_initial = circuit_input.copy()
    ###########################
    # 2. Unoptimization
    lap_time1 = time.time()
    v_original = operation_to_zero_state(circuit_initial)

    ###########################
    circuit_unopted, unopt_level = unoptimization(circuit_initial, iteration=Iteration, pair_select_method=pair_select_method)
    ###########################

    lap_time2 = time.time()
    ###########################
    # 3. Conversion
    circuit_qiskit, circuit_qiskit_with_unopt, circuit_tket, circuit_tket_with_unopt = circ_conversion(
        circuit_initial, circuit_unopted, v_original)
    lap_time3 = time.time()
    ###########################
    # 4. Compiler benchmark
    # qiskit
    df_qiskit = qiskit_comp(
        circuit_qiskit, circuit_qiskit_with_unopt, v_original)
    lap_time4 = time.time()

    # tket
    df_tket = tket_comp(circuit_tket, circuit_tket_with_unopt, v_original)
    ed_time = time.time()
    ###########################
    # 5. Result
    # time_result
    time_results = {'Unoptimization': lap_time2 - lap_time1,
                    'Conversion': lap_time3 - lap_time2,
                    'Qiskit_opt': lap_time4 - lap_time3,
                    'Tket_opt': ed_time - lap_time4,
                    'Whole_operation': ed_time - st_time}

    df_time = pd.DataFrame([time_results])

    return df_qiskit, df_tket, df_time, unopt_level