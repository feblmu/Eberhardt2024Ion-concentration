#!/home/eberhardt/anaconda2/envs/spineGeometryEnv/bin/python

# run simulation with standard parameter set

# run with: "ipython run.py -run_id" where run id is an integer setting the parameters 

import sys
sys.path.append('./../')
import spineSimulator


print('Running Python interpreter {_}.'.format(_=sys.executable))

if __name__ == '__main__':
    import sys
    print(sys.argv)
    spineSimulator.run(run_id = int(sys.argv[1]))
