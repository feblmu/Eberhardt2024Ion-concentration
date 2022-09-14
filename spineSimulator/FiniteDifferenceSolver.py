import numpy as np
import dbm.dumb as dbm
from constants import parameter_sets

class FiniteDifferenceSolver:

    def __init__(self,
        t,  # points on grid in time, has to start with 0
        x,  # points on grid in space, has to start with 0
        a,  # radius of cylindersegment along x
        bnds=[[0],[15.e-12],[-0.07]],  # time when to apply new bondary conditions, neumann current bundary, dirichlet bnd
        input_type='const',  # type of input current const electrod current or voltage and concentration dependent ion-channel current
        file_name = False,  # if file-name if provided dependent varibles get saved every 0.05 ms
        write_interval = 0.00001,
        parameter_set = 'standard',
        ):
        
        # load parmeters for simulation   
        self.parameter_set = parameter_set
        self.load_parameters(parameter_sets[self.parameter_set])
        
        # TODO: improve t&x argumnets in init or shift automatically
        if t[0] != 0. or x[0] != 0.:
            raise AssertionError('x and t have to start with 0')
            
        if input_type == 'const':
            self.input_type = input_type
            self.get_input_current = self.constant_input_current
            
        elif input_type == 'ion-channel':
            self.input_type = input_type
            self.get_input_current = self.variable_input_current
        else:
            self.input_type = 'const'
            print('ATTENTION: Input type set to "const". input_type argument must be "const" or "ion-channel".')
        
                
        # grid scale
        self.delta_x = (x[1] - x[0]) * self.scale_space
        self.delta_t = (t[1] - t[0]) * self.scale_time

        # number of grid points 
        # add two points to space grid (one left, one right) to realize boundary conditions later
        self.nx = np.size(x) + 2 # total number of points in x
        self.nt = np.size(t) # total number of time steps

        # independent variables space and time (x, t)
        self.x = np.zeros(self.nx)
        self.x[1:-1] = x
        self.x[0] = self.x[1] + x[0] - x[1]
        self.x[-1] = self.x[-2] + x[1] - x[0]
        self.x = self.x * self.scale_space
        
        self.t = t * self.scale_time # time, independent variable [t] = s
        
        # current time index
        self.t_i = 0  
        
        # shape parameter radius at x_i
        self.a = np.zeros(self.nx)
        self.a[1:-1] = a
        self.a[0] = self.a[1]
        self.a[-1] = self.a[-2]
        self.a = self.a * self.scale_space # radius at point x [a] = m
        
        # initialize dependent variables
        self.phi = np.full((self.nx), self.const_phi_rest)  # electric potential
        self.c_Na = np.full((self.nx), self.const_c_Na_rest)  # sodium concentration
        self.c_K = np.full((self.nx), self.const_c_K_rest)  # sodium concentration
        self.c_Cl = np.full((self.nx), self.const_c_Cl_rest)  # sodium concentration
        # background concentration will lead to resting potential on membrane capacitor
        self.c_background = (  
            2. / self.a * self.const_phi_rest * self.const_c_m / self.const_q  - 
            self.const_z_Na *  self.c_Na - 
            self.const_z_K * self.c_K - 
            self.const_z_Cl * self.c_Cl
        ) / self.const_z_background
        
        # initialize other variables that vary over time
        self.r_e_Na = np.zeros((self.nx)) # intracellular electrical resistance [r_e] = Ohm m
        self.r_e_K  = np.zeros((self.nx)) # intracellular electrical resistance [r_e] = Ohm m
        self.r_e_Cl = np.zeros((self.nx)) # intracellular electrical resistance [r_e] = Ohm m
        self.r_e = np.zeros((self.nx)) # intracellular electrical resistance [r_e] = Ohm m

        # compute initial values of r_e_Na, r_e_Cl, r_e_K, r_e along x
        self.update_electrical_resistance()

        # apply boundary conditions
        # ATTENTION appliy boundary conditions after all other variable are computed for the current time step
        # Neumann boundary at x=0 to model input current, gets computed from input conductance
        # Dirichlet boundary; set potential in dendrite at x=x_max to model large reservoir in dendrites and bAPs
        self.new_bnd_times_as_index = [np.sum( (t_ * self.scale_time) >  self.t) for t_ in bnds[0]]  # list of time points as time indices when to apply new boundary conditions
        self.neumann_bnd_conductances = [con / self.scale_resistance for con in bnds[1]]  # list of input conductances to compute neumann boundary conditions
        self.dirichlet_bnd_potentials = [phi_bnd * self.scale_voltage for phi_bnd in bnds[2]]  # list of membrane potentials in end segment in dendrite

        # TODO: this gets repeadted in solve() -> improve
        self.input_conductance = self.neumann_bnd_conductances[0] 
        self.phi_dendrite = self.dirichlet_bnd_potentials[0] 
        self.apply_boundary_conditions()
        
        print('Strength of input current: {i_} pA'.format(i_=self.get_input_current()/self.scale_current/1.e-12))
        
        self.file_name = file_name
        
        if self.file_name!=False :
            self.write_interval = write_interval * self.scale_time
            # number of timesteps between writes
            self.write_delta_t_i = int(self.write_interval / self.delta_t)
            print('Writing results to file every {ti} steps.'.format(ti=self.write_delta_t_i))
            print('Writing results to file every {t} seconds.'.format(t=self.write_delta_t_i/self.scale_time*self.delta_t))
            self.db = dbm.open(self.file_name, 'n')
            self.db['t'] = (self.t[::self.write_delta_t_i]/self.scale_time).tobytes()
            self.db['x'] = (self.x/self.scale_space).tobytes()
            self.db['radius'] = (self.a/self.scale_space).tobytes()
            self.db['parameters'] = self.parameter_set
            
            self.n_writes = 0  # counter for number of writes to file
            self.write_results()
        
        if True == 0:
            print('phi:' ,self.phi / self.scale_voltage)
            print('c_Na: ', self.c_Na / self.scale_concentration)
            print('c_K: ', self.c_K / self.scale_concentration)
            print('c_Cl: ', self.c_Cl / self.scale_concentration)
            print('Nernst Na: ', self.phi_nernst_Na / self.scale_voltage)
            print('I_syn: ', self.i_syn_AMPA / self.scale_current)
            print('I_m: ', self.i_m / self.scale_current)
            print('r_e: [Ohm m]', self.r_e/self.scale_resistance/self.scale_space)
    
    def load_parameters(self, constants):
        """
        constants: dict with all constants and parameters. Can be called with a different dict to updated parameters during a simulation. Or to load results from a previous simulation
        """
        
        # constants
        self.const_k_B = constants['const_k_B']  # Boltzmann constant [k_B] = m^2 kg s^-2 K^-1
        self.const_e = constants['const_e']  # elementary charge [e] = C
        self.const_N_A = constants['const_N_A']  # Avogadro constant [N_A] = mol^-1
        self.const_z_Na = constants['const_z_Na']  # charge number of sodium
        self.const_z_K = constants['const_z_K'] # charge number of potassium
        self.const_z_Cl = constants['const_z_Cl']  # charge number of chloride
        self.const_z_background = constants['const_z_background']  # charge number of background concentrations

        # multiplicative scaling factors to make equations dimensionless
        self.scale_voltage = constants['scale_voltage']  
        self.scale_space = constants['scale_space'] 
        self.scale_time = constants['scale_time'] 
        self.scale_concentration = constants['scale_concentration']
        self.scale_diffusion = constants['scale_diffusion']
        self.scale_charge = constants['scale_charge']
        self.scale_current = constants['scale_current']
        self.scale_resistance = constants['scale_resistance']
        self.scale_capacitance = constants['scale_capacitance']

        # scaled constants
        self.const_q = constants['const_q']  # scaled elementary charge
        self.const_k = constants['const_k']  # scaled boltzmann constant

        # spine parameters, all set in SI-units and then made unit-less by scaling factors
        self.const_D_Na = constants['const_D_Na'] # Diffusion Sodium 
        self.const_D_K = constants['const_D_K']  # Diffusion potassium 
        self.const_D_Cl = constants['const_D_Cl']  # Diffusion chloride  
        # self.const_r_m_Na = constants['const_r_m_Na'] # membrane resistance for sodium current
        # self.const_r_m_K = constants['const_r_m_K'] # membrane reistance for potassium current 
        # self.const_r_m_Cl = constants['const_r_m_Cl'] # membrane reistance for chloride current 
        self.const_c_m = constants['const_c_m'] # membrane capacitance 
        self.const_T = constants['const_T']  # Temperature 
        self.const_phi_rest = constants['const_phi_rest']  # resting membrane potential
        self.const_driving_voltage = constants['const_driving_voltage'] # driving voltage for constant input current
        self.const_c_Na_extracell = constants['const_c_Na_extracell']  # mmol 
        self.const_c_K_extracell =  constants['const_c_K_extracell'] # mmol 
        self.const_c_Cl_extracell = constants['const_c_Cl_extracell'] # mmol 
        self.const_c_Na_rest = constants['const_c_Na_rest']  # intracell. sodium concentration at rest
        self.const_c_K_rest = constants['const_c_K_rest']  # intracell. sodium concentration at rest
        self.const_c_Cl_rest = constants['const_c_Cl_rest']  # intracell. sodium concentration at rest   
    
    def apply_boundary_conditions(self,):
        """
        apply all boundary conditions
        """
        self.apply_neumann_boundary(self.phi, dydx=self.electric_potential_neumann_boundary())
        self.apply_neumann_boundary(self.c_Na, dydx=self.sodium_concentration_neumann_boundary())
        self.apply_neumann_boundary(self.c_K, dydx=0.,)
        self.apply_neumann_boundary(self.c_Cl, dydx=0.,)
        self.apply_dirichlet_boundary(self.phi, self.phi_dendrite)
        self.apply_dirichlet_boundary(self.c_Na, self.const_c_Na_rest)
        self.apply_dirichlet_boundary(self.c_K, self.const_c_K_rest)
        self.apply_dirichlet_boundary(self.c_Cl, self.const_c_Cl_rest)
        
    def apply_neumann_boundary(self, variable, dydx):
        """
        apply neumann boundary condition on potential or concentration (spatial derivative)
        neumann boundary conditions are applied at the left side (x=0) where synapse is located to model input current
        variable: reference to dependent variable on which to apply bnd (phi, c_i)
        dydx: value of derivativ at boundary (dydx=0 means reflecting or no-flux boundary)
        return: None
        """       
        variable[0] = variable[1] - dydx
    
    def apply_dirichlet_boundary(self, variable, yD):
        """
        apply dirichlet boundary condition on potential or concentration variable
        dirichlet boundary conditions are applied on the right side (x=xmax) where dendrite is located
        variable: reference to dependent variable on which to apply bnd (phi, c_i)
        yD: value of variable at boundary
        return: None
        """
        variable[-1] = yD
    
    def write_results(self):
        print('Writing results at {t} ms'.format(t=(self.t[self.t_i]/self.scale_time*1000)))
        self.db['phi'+str(self.n_writes)] = (self.phi/self.scale_voltage).tobytes()
        self.db['c_Na'+str(self.n_writes)] = (self.c_Na/self.scale_concentration).tobytes()
        self.db['c_K'+str(self.n_writes)] = (self.c_K/self.scale_concentration).tobytes()
        self.db['c_Cl'+str(self.n_writes)] = (self.c_Cl/self.scale_concentration).tobytes()
        self.n_writes = self.n_writes + 1
    
    def update_electrical_resistance(self,):
        """
        compute electrical resistance of intracellular space from constants and charge concentrations
        at time t_i
        """
        self.r_e_Na = self.const_k * self.const_T / self.const_q**2 / (
            self.const_D_Na * self.const_z_Na**2 * self.c_Na)
        self.r_e_K = self.const_k * self.const_T / self.const_q**2 / (
            self.const_D_K  * self.const_z_K**2  * self.c_K)
        self.r_e_Cl = self.const_k * self.const_T / self.const_q**2 / (
            self.const_D_Cl * self.const_z_Cl**2 * self.c_Cl)
        self.r_e = 1./ ( 1./self.r_e_Na + 1./self.r_e_K + 1./self.r_e_Cl )
        
    def g_k(self, r_e):
        """
        electrical conductivity coefficient
        electrical resistance of one particular ion type (e.g. r_e_Na) or total r_e
        """
        return np.square(self.a) / r_e
        
    def h_k(self, zD):
        """
        chemical conductivity coefficient
        zD: product of charge number z_i and Diffusion coefficient D_i of a particular ion-type i 
        """
        return np.square(self.a) * zD * self.const_q
        
    def gamma(self):
        return 1. / ( 2. * self.a * self.const_c_m)
    
    def delta_k(self, z):
        """
        maps a concentration per time chage to change of charges per time per unit length
        z: charge number z_i of a particular ion-type i 
        """
        return 1. / (np.square(self.a) * z * self.const_q)
        
    def d2fdx2(self, var, coeff):
        """
        class variable phi or c_k, 
        t_i: time index
        coeff: funciton or method
        """
        outflow = (coeff[2:] + coeff[1:-1])/2.*(var[2:] - var[1:-1]) / self.delta_x**2
        inflow = (coeff[0:-2] + coeff[1:-1])/2.*(var[1:-1] - var[:-2]) / self.delta_x**2
        return outflow - inflow
        
    def solve(self, method='explicit'):
        """
        solve system for all times t_0 to t_{M-1} 
               
        compute electric potential and concentrations at timestep t_{i+1}
        
        t_i: index of current timestep
        method: method to compute dependent variables at time ti+1
        
        return: None
        
        """
        if method == 'explicit':
                step_forward = self.explicit_step
        elif method == 'implicit':
            pass
            # step_forward = self.implicit_step
            # TODO implement implicit solver to increase spatial reoslution
        
        
        change_bnd_indices = self.new_bnd_times_as_index + [self.nt]
        
        for i, t_i_start in enumerate(change_bnd_indices[:-1]):
            # apply new boudary conditions
            self.input_conductance = self.neumann_bnd_conductances[i] 
            self.phi_dendrite = self.dirichlet_bnd_potentials[i] 
            for t_i in range(t_i_start, change_bnd_indices[i+1]):
            
                # update concentrations and voltage from t_i to t_{i+1}
                step_forward()
                
                # update time from t=t_i to t=t_{i+1}
                self.t_i += 1   
                
                # write results if wanted 
                if self.file_name:
                    if self.t_i % self.write_delta_t_i == 0:
                        self.write_results()
            
        if self.file_name!=False:
            self.db.close()
        
    def explicit_step(self):
        """
        implementation of explicit solver
        """
        #import time
        #t1 = time.time()
        """
        delta_phi = (
            self.gamma()[1:-1] * self.d2fdx2(self.phi, self.g_k(self.r_e))
            + 
            self.gamma()[1:-1] * self.d2fdx2(self.c_Na, self.h_k(self.const_z_Na * self.const_D_Na))
            +
            self.gamma()[1:-1] * self.d2fdx2(self.c_K , self.h_k(self.const_z_K * self.const_D_K))
            +
            self.gamma()[1:-1] * self.d2fdx2(self.c_Cl, self.h_k(self.const_z_Cl * self.const_D_Cl))
            ) 
            
        self.phi[1:-1]  = self.phi[1:-1]  + self.delta_t * delta_phi     
        """


        delta_c_Na = (
            self.delta_k(self.const_z_Na)[1:-1] * self.d2fdx2(self.phi, self.g_k(self.r_e_Na))
            + 
            self.delta_k(self.const_z_Na)[1:-1] * self.d2fdx2(self.c_Na, self.h_k(self.const_z_Na * self.const_D_Na))
            )

        delta_c_K = (
            self.delta_k(self.const_z_K)[1:-1] * self.d2fdx2(self.phi, self.g_k(self.r_e_K))
            + 
            self.delta_k(self.const_z_K)[1:-1] * self.d2fdx2(self.c_K, self.h_k(self.const_z_K * self.const_D_K))
            )

        delta_c_Cl = (
            self.delta_k(self.const_z_Cl)[1:-1] * self.d2fdx2(self.phi, self.g_k(self.r_e_Cl))
            + 
            self.delta_k(self.const_z_Cl)[1:-1] * self.d2fdx2(self.c_Cl, self.h_k(self.const_z_Cl * self.const_D_Cl))
            )

        # update dependent variables t+1
        self.c_Na[1:-1] = self.c_Na[1:-1] + self.delta_t * delta_c_Na
        self.c_K[1:-1]  = self.c_K[1:-1]  + self.delta_t * delta_c_K
        self.c_Cl[1:-1] = self.c_Cl[1:-1] + self.delta_t * delta_c_Cl
        
        self.phi = self.const_q * (
            self.const_z_background * self.c_background + 
            self.const_z_Na * self.c_Na +
            self.const_z_K * self.c_K + 
            self.const_z_Cl * self.c_Cl
        ) / self.const_c_m * self.a / 2.

        # update other variables to t+1
        self.update_electrical_resistance()

        # boundary contidions
        self.apply_boundary_conditions()
        
    def electric_potential_neumann_boundary(self,):
        """
        compute neumann boundary condition for the electrical potential
        """
        
        dPhi_dx = - self.get_input_current() * self.r_e[1] / np.pi / self.a[1]**2
        return dPhi_dx
        
    def sodium_concentration_neumann_boundary(self, ):
        """
        compute neumann boundary condition for the electrical potential for sodium concentration
        """
        dcNa_dx = -1. * self.get_input_current() / self.const_D_Na / self.const_q / self.const_z_Na / np.pi / self.a[1]**2
        return dcNa_dx
        
    def constant_input_current(self):
        """
        return constant input current for a driving voltage of 100mV
        current = 100mV * self.input_conductance
        """
        driving_voltage = self.const_driving_voltage
        current = driving_voltage * self.input_conductance # input current
        return current
    
    def variable_input_current(self):
        """
        Return input current for a variable driving voltage.       
        Compute driving voltage as difference between sodium nernst potential
        and current membrane potential. This is voltage and concentration dependent
        """
        V_T = self.const_k * self.const_T / self.const_q  #thermal voltage
        phi_nernst_Na = -V_T / self.const_z_Na * np.log(self.c_Na[1] / self.const_c_Na_extracell)
        driving_voltage = phi_nernst_Na - self.phi[1]  
        current = driving_voltage * self.input_conductance # input current
        return current
    
