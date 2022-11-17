from .FiniteDifferenceSolver import FiniteDifferenceSolver
from .constants import parameter_sets
from .plot import figure_head_overview

def get_a(n_h, n_hnj, n_n, n_ndj, n_d, nx, a_h, a_n, a_d=400.e-9):
    import numpy as np
    a = np.zeros(nx)
    a[:n_h] = a_h
    a[n_h+n_hnj : n_h+n_hnj+n_n] = a_n
    a[n_h+n_hnj+n_n+n_ndj : n_h+n_hnj+n_n+n_ndj+n_d] = a_d
    a[n_h : n_h+n_hnj]= np.linspace(a_h, a_n, n_hnj+2, endpoint=True)[1:-1]
    a[n_h+n_hnj+n_n : n_h+n_hnj+n_n+n_ndj] = np.linspace(a_n, a_d, n_ndj+2, endpoint=True)[1:-1]
    return a
    
def get_x(L, nx):   
    import numpy as np
    x = np.linspace(0,L,nx)
    return x
    
    
def get_t(T, timestep):
    import numpy as np
    nt = int(T/timestep)
    t = np.linspace(0., T, nt+1)
    return t

def run(run_id):
    import pandas, time
    df = pandas.read_excel('./../simulation_parameters.xls')
    results_file = './../../simulation_results/experiment_' + str(run_id)

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
    a_d = df.loc[10, run_id]
    input_type = df.loc[11, run_id]
    constants = df.loc[12, run_id]
    write_interval = df.loc[13, run_id]
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

    spine = FiniteDifferenceSolver(
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
    print('time taken {tsim} seconds for {nt} steps.'.format(nt=len(t), tsim=end-start))

    ########################################################
