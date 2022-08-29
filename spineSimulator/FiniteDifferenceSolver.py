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
    # scale_current = 1.e2  # 10 milliampere
    #scale_charge = 1. *scale_time *scale_current  # unit charge is unit_current * unit_time
    scale_resistance = scale_space**2/scale_time**3/scale_current**2  # 1/10^18 OHM
    scale_capacitance = scale_charge / scale_voltage #scale_current**2 * scale_time**4 / scale_space**2
    # scale_mass = 1. # maps kg -> not needed
    # scale_temperature = 1.
    
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
        t,  # points on grid in time
        x,  # points on grid in space
        a,  # radius of cylindersegment along x
        file_name = False  # if file-name if provided dependent varibles get saved every 0.05 ms
        ):
        
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
        
        # Neumann boundary at x=0
        # Dirichlet boundary at x=x_max
        # are satisfied automaticially by initialization of variables
        
        # initialize other variables that vary over time
        self.r_e_Na = np.zeros((self.nx)) # intracellular electrical resistance [r_e] = Ohm m
        self.r_e_K  = np.zeros((self.nx)) # intracellular electrical resistance [r_e] = Ohm m
        self.r_e_Cl = np.zeros((self.nx)) # intracellular electrical resistance [r_e] = Ohm m
        self.r_e = np.zeros((self.nx)) # intracellular electrical resistance [r_e] = Ohm m
        self.phi_nernst_Na = np.zeros((self.nx))  # nernst potential for sodium
        self.phi_nernst_K  = np.zeros((self.nx))  # nernst potential for potassium
        self.phi_nernst_Cl = np.zeros((self.nx))  # nernst potential for chloride
        self.i_syn_AMPA = np.zeros((self.nx))  # synpatic AMPA-current
        self.i_m_Na = np.zeros((self.nx))  # membrane leakage current of sodium
        self.i_m_K  = np.zeros((self.nx))  # membrane leakage current of potassium
        self.i_m_Cl = np.zeros((self.nx))  # membrane leakage current of chloride
        self.i_m = np.zeros((self.nx))  # total membrane leakage current
        
        # compute initial values of r_e_Na, r_e_Cl, r_e_K, r_e along x
        self.update_electrical_resistance()
        # compute initial value of synaptic AMPA current along x
        self.update_synaptic_AMPA_current()
        
        # compute inital values of phi_nernst_Na, phi_nernst_K, phi_Nernst_Cl along x
        ##self.update_nernst_potentials()
        # compute inital values of i_m_Na, i_m_K, i_m_Cl, i_m (membrane leakage current) along x
        ##self.update_membrane_current()
        
        self.file_name = file_name
        self.write_interval = 0.00001
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
        
    def write_results(self):
        print('Writing results at {t} ms'.format( t=(self.t[self.t_i]/self.scale_time*1000) ))
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
        
    def update_membrane_current(self):
        """
        ohmic membrane current
        drivng voltage is the difference between the current membrane potential and the
        current resting potential of the different ions (nernst potentials)
        """
        self.i_m_Na = (self.phi - self.phi_nernst_Na) / self.const_r_m_Na
        self.i_m_K  = (self.phi - self.phi_nernst_K) / self.const_r_m_K
        self.i_m_Cl = (self.phi - self.phi_nernst_Cl) / self.const_r_m_Cl
        self.i_m = self.i_m_Na + self.i_m_K + self.i_m_Cl
        
    def update_nernst_potentials(self):
        """
        compute resting potential. This is the nernst-potential of the 
        see eqs: 5.1 & 5.4 in dyan & abbott
        """
        V_T = self.const_k * self.const_T / self.const_q
        self.phi_nernst_Na = -V_T / self.const_z_Na * np.log(self.c_Na / self.const_c_Na_extracell)
        self.phi_nernst_K  = -V_T / self.const_z_K  * np.log(self.c_K  / self.const_c_K_extracell )
        self.phi_nernst_Cl = -V_T / self.const_z_Cl * np.log(self.c_Cl / self.const_c_Cl_extracell)
                
    def update_synaptic_AMPA_current(self, duration=0.01, ampl=15.e-12):
        """
        AMPA current is assumed to be purely sodium current
        AMPA enters at left side x=0 at segment with index 1
        (index 0 segment is there to implement boundary conditions)
        simply model with constant synaptic conductivity
        
        duration: lenght synapse opening in seconds
        ampl: current aplitude at 100 mV driving potential in Ampere
        """
        input_segments = 1
        current_time = self.t[self.t_i]
        segment_surface = 2. * np.pi * self.a[1:input_segments+1] * (self.x[input_segments+1] - self.x[1])
        current_density = ampl*self.scale_current/segment_surface
        
        if current_time <= (duration*self.scale_time):
            self.i_syn_AMPA[1:input_segments+1] = current_density
        else: 
            self.i_syn_AMPA[:] = 0.
        
    def g_k(self, ion_species):
        """
        electrical conductivity coefficient
        """
        if ion_species == 'Na':
            r_e = self.r_e_Na
        elif ion_species == 'K':
            r_e = self.r_e_K
        elif ion_species == 'Cl':
            r_e = self.r_e_Cl
        elif ion_species == 'all':
            r_e = self.r_e
        else: 
            raise AssertionError( 'invalid ion_species argument in method g_k' )
        return np.square(self.a) / r_e
        
    def h_k(self, ion_species):
        """
        chemical conductivity coefficient
        """
        if ion_species == 'Na':
            zD = self.const_z_Na * self.const_D_Na
        elif ion_species == 'K':
            zD = self.const_z_K * self.const_D_Na
        elif ion_species == 'Cl':
            zD = self.const_z_Cl * self.const_D_Na
        else: 
            raise AssertionError( 'invalid ion_species argument in method h_k' )
        return np.square(self.a) * zD * self.const_q
        
    def gamma(self):
        return 1. / ( 2. * self.a * self.const_c_m)
        
    def alpha(self):
        """
        multiplicative factor of synapic current of voltage equation
        """
        return np.full((self.nx), 1. / self.const_c_m)
    
    def delta_k(self, ion_species):
        """
        maps a concentration per time chage to change of charges per time per unit length
        """
        if ion_species == 'Na':
            z = self.const_z_Na
        elif ion_species == 'K':
            z = self.const_z_K
        elif ion_species == 'Cl':
            z = self.const_z_Cl
        else: 
            raise AssertionError( 'invalid ion_species argument in method delta_k' )
        return 1. / (np.square(self.a) * z * self.const_q)
        
    def beta_k(self, ion_species):
        if ion_species == 'Na':
            z = self.const_z_Na
        elif ion_species == 'K':
            z = self.const_z_K
        elif ion_species == 'Cl':
            z = self.const_z_Cl
        else: 
            raise AssertionError( 'invalid ion_species argument in method delta_k' )
        return 2. / (self.a * z * self.const_q)
        
    def d2fdx2(self, var, coeff):
        """
        class variable phi or c_k, 
        t_i: time index
        coeff: funciton or method
        
        """
        outflow = (coeff[2:] + coeff[1:-1])/2.*(var[2:] - var[1:-1]) / self.delta_x**2
        inflow = (coeff[0:-2] + coeff[1:-1])/2.*(var[1:-1] - var[:-2]) / self.delta_x**2
        return outflow - inflow
        
    def solve(self, method='leapfrog'):
        """
        solve system for all times t_0 to t_{M-1}
        """
        for t_i in range(self.nt):
            self.step_forward(method=method,)
            if t_i%100000==0:
                print(t_i, ' of ', self.nt)
            
        if self.file_name!=False:
            self.save_results()
    
    def step_forward(self, method='leapfrog',):
        """
        compute electric potential and concentrations at timestep t_{i+1}
        
        t_i: index of current timestep
        method: method to compute dependent variables at time ti+1
        
        return: None
        """
        # update concentrations and voltage from t_i to t_{i+1}
        if method == 'leapfrog':
            self.leapfrog()
        elif method == 'crank-nicolson':
            self.crank_nicolson()
        
        # update time from t=t_i to t=t_{i+1}
        self.t_i += 1   
        
        # write results if wanted 
        if self.file_name:
            if self.t_i % self.write_delta_t_i == 0:
                self.write_results()
            
        
    def leapfrog(self):
        """
        implementation of leapfrog algorithm
        """
        
        delta_phi = (
            self.gamma()[1:-1] * self.d2fdx2(self.phi, self.g_k('all'))
            + 
            self.gamma()[1:-1] * self.d2fdx2(self.c_Na, self.h_k('Na'))
            +
            self.gamma()[1:-1] * self.d2fdx2(self.c_K , self.h_k('K' ))
            +
            self.gamma()[1:-1] * self.d2fdx2(self.c_Cl, self.h_k('Cl'))
            + 
            self.alpha()[1:-1] * self.i_syn_AMPA[1:-1]
            ) 
            
        
        
        delta_c_Na = (
            self.delta_k('Na')[1:-1] * self.d2fdx2(self.phi, self.g_k('Na'))
            + 
            self.delta_k('Na')[1:-1] * self.d2fdx2(self.c_Na, self.h_k('Na'))
            +
            self.beta_k('Na')[1:-1] * self.i_syn_AMPA[1:-1]
            )
            
        delta_c_K = (
            self.delta_k('K')[1:-1] * self.d2fdx2(self.phi, self.g_k('K'))
            + 
            self.delta_k('K')[1:-1] * self.d2fdx2(self.c_K, self.h_k('K'))
            )
 
        delta_c_Cl = (
            self.delta_k('Cl')[1:-1] * self.d2fdx2(self.phi, self.g_k('Cl'))
            + 
            self.delta_k('Cl')[1:-1] * self.d2fdx2(self.c_Cl, self.h_k('Cl'))
            )
        
        #print('concentration delta q: ', (delta_c_Na + delta_c_K - delta_c_Cl) * self.const_q *np.square(self.a[1:-1]))
        #print('capactiance delta q: ', (delta_phi * 2. *self.a[1:-1] *self.const_c_m))
        
        # update dependent variables t+1
        self.phi[1:-1]  = self.phi[1:-1]  + self.delta_t * delta_phi
        self.c_Na[1:-1] = self.c_Na[1:-1] + self.delta_t * delta_c_Na
        self.c_K[1:-1]  = self.c_K[1:-1]  + self.delta_t * delta_c_K
        self.c_Cl[1:-1] = self.c_Cl[1:-1] + self.delta_t * delta_c_Cl
        
        # boundary contidions
        # dPhi/dx = 0 at x=0 (Neumann boundary)
        # Phi(x) = Phi_rest at x=x_max (Dirichlet boundary)
        self.phi[0] = self.phi[1]
        self.c_Na[0] = self.c_Na[1]
        self.c_K[0] = self.c_K[1]
        self.c_Cl[0] = self.c_Cl[1]
        self.phi[-1] = self.const_phi_rest
        self.c_Na[-1] = self.const_c_Na_rest
        self.c_K[-1] = self.const_c_K_rest
        self.c_Cl[-1] = self.const_c_Cl_rest
      
        
        # update other variables to t+1
        self.update_electrical_resistance()
        ## self.update_nernst_potentials()
        self.update_synaptic_AMPA_current()
        ## self.update_membrane_current()
        
        
    
    
