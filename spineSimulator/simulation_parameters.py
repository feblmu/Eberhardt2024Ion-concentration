import numpy as np

################################
# simulation parameters
################################

################################
# t-grid
T1 = 20.e-3
timestep1 = 1.e-10  # use at most 100 picoseconds for explicit solver, but need to be smaller if x-resolution gets higher
nt1 = int(T1/timestep1)
t1 = np.linspace(0., T1, nt1+1)

T2 = 0.01e-3
timestep2 = 1.e-10  
nt2 = int(T2/timestep2)
t2 = np.linspace(0., T2, nt2+1)

################################
# x-grid
L1 = 1.3e-6
nh1, nhnj1, nn1, nndj1, nd1 = 4, 1, 4, 1, 4
nx1 = nh1 + nhnj1 + nn1 + nndj1 + nd1
x1 = np.linspace(0,L1,nx1)


################################
# standard spine shape
ah1, an1, ad1 = 250.e-9, 35.e-9, 400.e-9 
a1 = np.zeros(nx1)
a1[:nh1]=ah1
a1[nh1+nhnj1:nh1+nhnj1+nn1] = an1
a1[nh1+nhnj1+nn1+nndj1: nh1+nhnj1+nn1+nndj1+nd1] = ad1
a1[nh1:nh1+nhnj1]= ah1 # np.linspace(ah, an, nhnj+2, endpoint=True)[1:-1]
a1[nh1+nhnj1+nn1 : nh1+nhnj1+nn1+nndj1] = an1 # np.linspace(an, ad, nndj+2, endpoint=True)[1:-1]

# lagache small
ah2, an2, ad2 = 150.e-9, 40.e-9, 400.e-9 
a2 = np.zeros(nx1)
a2[:nh1]=ah2
a2[nh1+nhnj1:nh1+nhnj1+nn1] = an2
a2[nh1+nhnj1+nn1+nndj1: nh1+nhnj1+nn1+nndj1+nd1] = ad2
a2[nh1:nh1+nhnj1]= ah2
a2[nh1+nhnj1+nn1 : nh1+nhnj1+nn1+nndj1] = an2

########################################

simulation_parameters = {
    'standard': (t1, x1, a1),
    'test': (t2, x1, a1),
    'lagache_small' : (t1, x1, a2),
}
