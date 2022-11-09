#!/home/eberhardt/anaconda2/envs/spineGeometryEnv/bin/python

# run simulation with standard parameter set

# run with: "ipython run.py -run_id" where run id is an integer setting the parameters 

import numpy as np
import pandas
import sys
sys.path.append('./spineSimulator/')
import FiniteDifferenceSolver
import time
from simulation_parameters import simulation_parameters

print('Running Python interpreter {_}.'.format(_=sys.executable))

def get_a(n_h, n_hnj, n_n, n_ndj, n_d, nx, a_h, a_n, a_d=400.e-9):
    a = np.zeros(nx)
    a[:n_h] = a_h
    a[n_h+n_hnj : n_h+n_hnj+n_n] = a_n
    a[n_h+n_hnj+n_n+n_ndj : n_h+n_hnj+n_n+n_ndj+n_d] = a_d
    a[n_h : n_h+n_hnj]= np.linspace(a_h, a_n, n_hnj+2, endpoint=True)[1:-1]
    a[n_h+n_hnj+n_n : n_h+n_hnj+n_n+n_ndj] = np.linspace(a_n, a_d, n_ndj+2, endpoint=True)[1:-1]
    
def get_x(L, nx):   
    x = np.linspace(0,L,nx)
    return x
    
    
def get_t(T, timestep):
    nt = int(T/timestep)
    t = np.linspace(0., T, nt+1)
    return t

def run():
    df = pandas.read_excel('./../simulation_parameters.xlsx')
    run_id = sys.argv[1]

    results_file = './../simulation_results/experiment_' + str(run_id)

    T = df.loc[0, run_id]
    delta_t = df.loc[1, run_id]
    L = df.loc[2, run_id]
    n_h = df.loc[3, run_id]
    n_hnj = df.loc[4, run_id]
    n_n = df.loc[5, run_id]
    n_ndj = df.loc[6, run_id]
    n_d = df.loc[7, run_id]
    nx = n_h + n_hnj + n_n + n_ndj + n_d
    a_h = df.loc[8, run_id]
    a_n = df.loc[9, run_id]
    a_d = df.loc[0, run_id]
    input_type = df.loc[11, run_id]
    constants = df.loc[12, run_id]
    write_dt = df.loc[13, run_id]
    bc_times = [float(v) for v in df.loc[14, run_id].split(',')]
    bc_vN_vals = [float(v) for v in df.loc[15, run_id].split(',')]
    bc_dir_vals = [float(v) for v in df.loc[16, run_id].split(',')]

    boundary_conditions = [bc_times,
              bc_vN_vals,
              bc_dir_vals]

    t = get_t(T, delta_t)
    x = get_x(L, nx)
    a = get_a(n_h, n_hnj, n_n, n_ndj, n_d, nx, a_h, a_n, a_d)
    ########################################################

    spine = FiniteDifferenceSolver.FiniteDifferenceSolver(
        t,x,a,
        bnds=boundary_conditions,
        input_type=input_type,
        file_name=results_file, 
        write_interval=write_interval,
        parameter_set = constants)

    start = time.time()
    spine.solve()
    end = time.time()
    print('time simulated [seconds]: ', T )
    print('time taken {tsim} seconds for {nt} steps.'.format(nt=nt, tsim=end-start))

    ########################################################

