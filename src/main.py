from unopt.benchmarking import qc_benchmark
import statistics
import pandas as pd
from tqdm import tqdm
import multiprocessing
from multiprocessing import Pool

def write_circuit_info(method, samples):
    with open('circuit_info.txt', 'w') as f:
        datalist = ['Random circuit\n', f'unopt_pair = {method}\n', f'Generated {samples} circuits for each n.']
        f.writelines(datalist)

def main():
    ########## Input ##########
    nqubits = 4 #initial number_of_qubits
    iteration = (nqubits)**2 #iteration_of_unopt
    samples = 3
    nqubits_max = 11
    method = 'concatenated' #'random' or 'concatenated'
    ###########################
    write_circuit_info(method, samples)
    df = pd.DataFrame()
    df_Time = pd.DataFrame()
    df_unopt_level = pd.DataFrame()
    col_names = ["unopt_level", "score_tk"]

    # benchmark
    for j in tqdm(range(nqubits, nqubits_max + 1)):
        a, b, c, d = [], [], [], []
        df_t = pd.DataFrame()
        print('---Circuit Info---')
        print('Nqubits: ',nqubits)
        print('Iteration: ',iteration)
        print('------------------')
        for _ in tqdm(range(samples)):
            result = qc_benchmark(Number_of_qubits=nqubits, Depth=nqubits, Iteration=iteration, pair_select_method=method)
            d_original = result[0].at['qiskit_u','Depth']
            d_unopt_qis = result[0].at['qiskit_v','Depth']
            d_opt_qis = result[0].at['qiskit_v_compiled','Depth']
            d_unopt_tk = result[1].at['tket_v','Depth']
            d_opt_tk = result[1].at['tket_v_compiled','Depth']
            a.append(d_unopt_qis / d_original) #r_unopt(Qiskit)
            b.append(d_opt_qis / d_original) #r_opt(Qiskit)
            c.append(d_unopt_tk / d_original) #r_unopt(Pytket)
            d.append(d_opt_tk / d_original) #r_opt(Pytket)
            data = [[result[3], d_opt_tk / d_original]] #[r_unopt(III), r_opt(Pytket)]
            df_u = pd.DataFrame(data, columns=col_names)
            df_unopt_level = pd.concat([df_unopt_level, df_u], axis=0)
            df_t = pd.concat([df_t, result[2]],axis=0)
        
        # calculate mean
        value_a = statistics.mean(a)
        value_b = statistics.mean(b)
        value_c = statistics.mean(c)
        value_d = statistics.mean(d)

        # calculate variance
        value_e = statistics.pvariance(a)
        value_f = statistics.pvariance(b)
        value_g = statistics.pvariance(c)
        value_h = statistics.pvariance(d)

        df_Time[j] = df_t.mean()
        df[j] = [value_a, value_b, value_c, value_d, value_e, value_f, value_g, value_h]

        # update circuit info
        nqubits += 1
        iteration = int((nqubits) ** 2)

        # save
        df.to_csv('result.csv')
        df_Time.to_csv('result_time.csv')
        df_unopt_level.to_csv('result_unopt_level.csv')

    print('Finished!')

def multi_main():
    ########## Input ##########
    nqubits = 4 #initial number_of_qubits
    iteration = (nqubits)**2 #iteration_of_unopt
    samples = 30
    nqubits_max = 11
    method = 'concatenated' #'random' or 'concatenated'
    ###########################
    write_circuit_info(method, samples)
    df = pd.DataFrame()
    df_unopt_level = pd.DataFrame()
    col_names = ["unopt_level", "score_tk"]


    # generate params
    params=[]
    for i in range(nqubits, nqubits_max + 1):
        for _ in range(samples):
            params.append((i, i, i**2, method)) # (Number_of_qubits, Depth, Iteration, pair_select_method)

    # benchmark
    with Pool(multiprocessing.cpu_count()) as pool:
        results = pool.starmap(qc_benchmark, params)

    # analyze data
    for i in range(nqubits_max - nqubits + 1):
        r_unopt_qis, r_opt_qis, r_unopt_tk, r_opt_tk = [], [], [], []
        for j in range(samples):
            k = samples*i+j
            d_original = results[k][0].at['qiskit_u','Depth']
            d_unopt_qis = results[k][0].at['qiskit_v','Depth']
            d_opt_qis = results[k][0].at['qiskit_v_compiled','Depth']
            d_unopt_tk = results[k][1].at['tket_v','Depth']
            d_opt_tk = results[k][1].at['tket_v_compiled','Depth']
            r_unopt_qis.append(d_unopt_qis / d_original) #r_unopt(Qiskit)
            r_opt_qis.append(d_opt_qis / d_original) #r_opt(Qiskit)
            r_unopt_tk.append(d_unopt_tk / d_original) #r_unopt(Pytket)
            r_opt_tk.append(d_opt_tk / d_original) #r_opt(Pytket)
            data = [[results[k][3], d_opt_tk / d_original]] #[r_unopt(III), r_opt(Pytket)]
            df_u = pd.DataFrame(data, columns=col_names)
            df_unopt_level = pd.concat([df_unopt_level, df_u], axis=0)
        
        # calculate mean
        value_a = statistics.mean(r_unopt_qis)
        value_b = statistics.mean(r_opt_qis)
        value_c = statistics.mean(r_unopt_tk)
        value_d = statistics.mean(r_opt_tk)

        # calculate variance
        value_e = statistics.pvariance(r_unopt_qis)
        value_f = statistics.pvariance(r_opt_qis)
        value_g = statistics.pvariance(r_unopt_tk)
        value_h = statistics.pvariance(r_opt_tk)

        df[i] = [value_a, value_b, value_c, value_d, value_e, value_f, value_g, value_h]

        # save
        df.to_csv('result.csv')
        df_unopt_level.to_csv('result_unopt_level.csv')
    print('Finished!')


if __name__ == "__main__":
    # Please choose one of the versions and run it.

    # Run the following line for the single-threaded version:
    main()

    # Run the following line for the multi-threaded version:
    print('MULTI')
    multi_main()