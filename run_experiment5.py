#!/home/eberhardt/anaconda2/envs/spineGeometryEnv/bin/python

# run simulation with standard parameter set

import numpy as np
import sys
sys.path.append('./spineSimulator/')
import FiniteDifferenceSolver
import time

print('Running Python interpreter {_}.'.format(_=sys.executable))

# x-grid
L = 1.5e-6
nh, nhnj, nn, nndj, nd = 4, 1, 4, 1, 4
nx = nh + nhnj + nn + nndj + nd
x = np.linspace(0,L,nx)

# t-grid
T = 20.e-3
timestep = 1.e-10  # use 100 picoseconds for explicit solver
nt = int(T/timestep)
t = np.linspace(0., T, nt+1)
print('timesteps {nt}'.format(nt=nt))

# spine shape
ah, an, ad = 250.e-9, 50.e-9, 400.e-9 
a = np.zeros(nx)
a[:nh]=ah
a[nh+nhnj:nh+nhnj+nn] = an
a[nh+nhnj+nn+nndj: nh+nhnj+nn+nndj+nd] = ad
a[nh:nh+nhnj]= np.linspace(ah, an, nhnj+2, endpoint=True)[1:-1]
a[nh+nhnj+nn : nh+nhnj+nn+nndj] = np.linspace(an, ad, nndj+2, endpoint=True)[1:-1]

results_file = './../simulation_results/experiment_5'
write_interval = 5.e-5
spine = FiniteDifferenceSolver.FiniteDifferenceSolver(
    t,x,a,
    bnds=[[0., 10.e-3,],
          [250.e-12, 0.e-12],
          [-0.07, -0.060,]],
    input_type='const',
    file_name=results_file, 
    write_interval=write_interval,
    parameter_set = 'standard')

# overwrite standard parameters
spine.const_D_Na = spine.const_D_K

start = time.time()
spine.solve()
end = time.time()
print('time simulated [seconds]: ', T )
print('time taken {tsim} seconds for {nt} steps.'.format(nt=nt, tsim=end-start))

