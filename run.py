#!/home/eberhardt/anaconda2/envs/spineGeometryEnv/bin/python

import numpy as np
import sys
sys.path.append('./spineSimulator/')
import FiniteDifferenceSolver
import time

print('Running Python interpreter {_}.'.format(_=sys.executable))

# x-grid
L = 1.e-6
nh, nhnj, nn, nndj, nd = 4, 1, 4, 1, 4
nx = nh + nhnj + nn + nndj + nd
x = np.linspace(0,L,nx)

# t-grid
T = 0.1e-3
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

results_file = './../../simulation_results/test_run_05092022.pcl'
write_interval = 1.e-7
spine = FiniteDifferenceSolver.FiniteDifferenceSolver(t,x,a,results_file, write_interval=write_interval)
start = time.time()
spine.solve()
end = time.time()
print('time simulated [seconds]: ', T )
print('time taken {tsim} seconds for {nt} steps.'.format(nt=nt, tsim=end-start))
