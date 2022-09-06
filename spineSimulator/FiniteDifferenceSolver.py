import numpy as np

class FiniteDifferenceSolver:
    
    # constants
    const_k_B = 1.381e-23  # Boltzmann constant [k_B] = m^2 kg s^-2 K^-1
    const_e = 1.602e-19  # elementary charge [e] = C
    const_N_A = 6.022e23  # Avogadro constant [N_A] = mol^-1
    const_z_Na = +1  # charge number of sodium
    const_z_K = +1 # charge number of potassium
    const_z_Cl = -1  # charge number of chloride

    # multiplicative scaling factors to make equations dimensionless
    scale_voltage = 1.e3  # 1./10 mV
    scale_space = 1.e8 # 1./10 nm
    scale_time = 1.e4 #  1./0.1 ms
    scale_concentration = const_N_A / scale_space**3. 
    scale_diffusion = scale_space**2. / scale_time  
    scale_charge = scale_space**2 / scale_time**2 / scale_voltage  # can be derived from thermal voltage V_T = k_B*t/e
    scale_current = scale_charge / scale_time
    scale_resistance = scale_space**2/scale_time**3/scale_current**2  # 1/10^18 OHM
    scale_capacitance = scale_charge / scale_voltage #scale_current**2 * scale_time**4 / scale_space**2
    # scale_mass = 1. # maps kg -> not needed
    # scale_temperature = 1.  # effectively no scaling just dimensionless
    
    # scaled constants
    const_q = const_e * scale_charge
    const_k = const_k_B /scale_time**2. *scale_space**2.
    
    # spine parameters, all set in SI-units and then made unit-less by scaling factors
    const_D_Na = 0.5000e-9 * scale_diffusion # Diffusion Sodium [D_Na] = m^2 s^-1
    const_D_K = 0.5000e-9 * scale_diffusion  # Diffusion potassium [D_K] = m^2 s^-1
    const_D_Cl = 0.5000e-9 * scale_diffusion  # Diffusion chloride [D_Cl] = m^2 s^-1
    const_r_m_Na = 1.* scale_resistance * scale_space**2 # membrane reistance for sodium current[r_m] = MOhm mm^2 = Ohm m^2
    const_r_m_K = 1.* scale_resistance * scale_space**2 # membrane reistance for potassium current [r_m] = MOhm mm^2 = Ohm m^2
    const_r_m_Cl = 1.* scale_resistance * scale_space**2 # membrane reistance for chloride current [r_m] = MOhm mm^2 = Ohm m^2
    const_c_m = 0.01*scale_capacitance/scale_space**2 # membrane capacitance [c_m] = F/m^2
    const_T = 310.0  # Temperature [T] = K
    const_c_Na_extracell = 145. * scale_concentration  # mmol 
    const_c_K_extracell =  5  * scale_concentration # mmol 
    const_c_Cl_extracell = 110. * scale_concentration # mmol 
    const_c_Na_rest = 10. * scale_concentration  # intracell. sodium concentration at rest
    const_c_K_rest = 140. * scale_concentration  # intracell. sodium concentration at rest
    const_c_Cl_rest = 10. * scale_concentration  # intracell. sodium concentration at rest
    const_phi_rest = -0.07 * scale_voltage  # resting membrane potential
    # for ion concentrations see: https://bionumbers.hms.harvard.edu/files/Comparison%20of%20ion%20concentrations%20inside%20and%20outside%20a%20typical%20mammalian%20cell.jpg
    # for membrane capacitance see: https://bionumbers.hms.harvard.edu/bionumber.aspx?&id=110759
    # membrane resistance see dyan & abbott p. 207
    
    def __init__(self,
        t,  # points on grid in time, has to start with 0
        x,  # points on grid in space, has to start with 0
        a,  # radius of cylindersegment along x
        bnds=[[0],[15.e-12],[-0.07]],  # time when to apply new bondary conditions, neumann current bundary, dirichlet bnd
        input_type='const',  # type of input current const electrod current or voltage and concentration dependent ion-channel current
        file_name = False,  # if file-name if provided dependent varibles get saved every 0.05 ms
        write_interval = 0.00001,
        ):
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
        self.write_interval = write_interval
        if self.file_name!=False :
            self.results = {'data':{}, 'params':{}}
            self.write_delta_t_i = int((self.write_interval * self.scale_time) / self.delta_t)
            print('Writing results to file every {ti} steps.'.format(ti=self.write_delta_t_i))
            print('Writing results to file every {t} seconds.'.format(t=self.write_delta_t_i/self.scale_time*self.delta_t))
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
        self.results['data'].update(
            {self.t[self.t_i]/self.scale_time: 
                {
                'phi': self.phi/self.scale_voltage,
                'c_Na': self.c_Na/self.scale_concentration,
                'c_K' : self.c_K/self.scale_concentration,
                'c_Cl': self.c_Cl/self.scale_concentration,
                }
            }
            )

    def save_results(self):
        self.results['params'].update({'t': self.t/self.scale_time})
        self.results['params'].update({'x': self.x/self.scale_space})
        self.results['params'].update({'radius': self.a/self.scale_space})
        self.results['params'].update({'parameter_set': 'standard'})
        
        import pickle
        pickle.dump(self.results, open(self.file_name, 'wb'))
    
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
            self.save_results()
        
    def explicit_step(self):
        """
        implementation of explicit solver
        """
        #import time
        #t1 = time.time()
        
        delta_phi = (
            self.gamma()[1:-1] * self.d2fdx2(self.phi, self.g_k(self.r_e))
            + 
            self.gamma()[1:-1] * self.d2fdx2(self.c_Na, self.h_k(self.const_z_Na * self.const_D_Na))
            +
            self.gamma()[1:-1] * self.d2fdx2(self.c_K , self.h_k(self.const_z_K * self.const_D_K))
            +
            self.gamma()[1:-1] * self.d2fdx2(self.c_Cl, self.h_k(self.const_z_Cl * self.const_D_Cl))
            ) 
            
        #t2 = time.time()
        #print('t2 -t1: ', (t2 - t1)/1.e-3)
        delta_c_Na = (
            self.delta_k(self.const_z_Na)[1:-1] * self.d2fdx2(self.phi, self.g_k(self.r_e_Na))
            + 
            self.delta_k(self.const_z_Na)[1:-1] * self.d2fdx2(self.c_Na, self.h_k(self.const_z_Na * self.const_D_Na))
            )
        #t3 = time.time()    
        #print('t3 -t2: ', (t3 - t2)/1.e-3)
        delta_c_K = (
            self.delta_k(self.const_z_K)[1:-1] * self.d2fdx2(self.phi, self.g_k(self.r_e_K))
            + 
            self.delta_k(self.const_z_K)[1:-1] * self.d2fdx2(self.c_K, self.h_k(self.const_z_K * self.const_D_K))
            )
        #t4 = time.time()
        #print('t4 -t3: ', (t4 - t3)/1.e-3)
        delta_c_Cl = (
            self.delta_k(self.const_z_Cl)[1:-1] * self.d2fdx2(self.phi, self.g_k(self.r_e_Cl))
            + 
            self.delta_k(self.const_z_Cl)[1:-1] * self.d2fdx2(self.c_Cl, self.h_k(self.const_z_Cl * self.const_D_Cl))
            )
        #t5 = time.time()
        #print('t5 -t4: ', (t5 - t3)/1.e-3)
        # update dependent variables t+1
        self.phi[1:-1]  = self.phi[1:-1]  + self.delta_t * delta_phi
        self.c_Na[1:-1] = self.c_Na[1:-1] + self.delta_t * delta_c_Na
        self.c_K[1:-1]  = self.c_K[1:-1]  + self.delta_t * delta_c_K
        self.c_Cl[1:-1] = self.c_Cl[1:-1] + self.delta_t * delta_c_Cl
        #t6 = time.time()
        #print('t6 -t5: ', (t6 - t5)/1.e-3)
        # update other variables to t+1
        self.update_electrical_resistance()
        #t7 = time.time()
        #print('t7 -t6: ', (t7 - t6)/1.e-3)
        # boundary contidions
        self.apply_boundary_conditions()
        
        #t8 = time.time()
        #print('t8 -t7: ', (t8 - t7)/1.e-3)
        #print('total time: ', (t8 - t1)/1.e-3)
        
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
        driving_voltage = 0.1 * self.scale_voltage
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
    
