#!/home/eberhardt/anaconda2/envs/spineGeometryEnv/bin/python

# run simulation with standard parameter set

import numpy as np
import sys
sys.path.append('./spineSimulator/')
import FiniteDifferenceSolver
import time
from simulation_parameters import simulation_parameters

print('Running Python interpreter {_}.'.format(_=sys.executable))

########################################################

t,x,a = simulation_parameters['lagache_small']
input_type = 'const'
constants = 'lagache'

boundary_conditions = [[0., 10.e-3,],
          [400.e-12, 0.e-12],
          [-0.07, -0.07,]]

results_file = './../simulation_results/experiment_8'
write_interval = 5.e-5

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
