import copy

# natural constants
const_k_B = 1.381e-23  # Boltzmann constant [k_B] = m^2 kg s^-2 K^-1
const_e = 1.602e-19  # elementary charge [e] = C
const_N_A = 6.022e23 # Avogadro constant [N_A] = mol^-1
const_z_Na = +1  # charge number of sodium
const_z_K = +1 # charge number of potassium
const_z_Cl = -1  # charge number of chloride
const_z_background = -1  # charge number of background concentrations

# multiplicative scaling factors to make equations dimensionless
scale_voltage = 1.e3  # 1./10 mV
scale_space = 1.e8 # 1./10 nm
scale_time = 1.e4 #  1./0.1 ms
scale_concentration = const_N_A / scale_space**3.
scale_diffusion = scale_space**2. / scale_time
scale_charge = scale_space**2 / scale_time**2 / scale_voltage  # can be derived from thermal voltage V_T : k_B*t/e
scale_current = scale_charge / scale_time
scale_resistance = scale_space**2/scale_time**3/scale_current**2  # 1/10^18 OHM
scale_capacitance = scale_charge / scale_voltage #scale_current**2 * scale_time**4 / scale_space**2
# scale_mass = 1. # maps kg -> not needed
# scale_temperature = 1.  # effectively no scaling just dimensionless

# spine parameters, all set in SI-units and then made unit-less by scaling factors
const_D_Na = 0.6500e-9  # Diffusion Sodium [D_Na] = m^2 s^-1
const_D_K = 1.000e-9   # Diffusion potassium [D_K] = m^2 s^-1
const_D_Cl = 1.0000e-9  # Diffusion chloride [D_Cl] = m^2 s^-1
const_r_m_Na = 1. # membrane resistance for sodium current[r_m] = MOhm mm^2 : Ohm m^2
const_r_m_K = 1. # membrane resistance for potassium current [r_m] = MOhm mm^2 = Ohm m^2
const_r_m_Cl = 1. # membrane resistance for chloride current [r_m] = MOhm mm^2 = Ohm m^2
const_c_m = 0.01 # membrane capacitance [c_m] = F/m^2
const_T = 310.0 # Temperature [T] = K
const_phi_rest = -0.07  # resting membrane potential
const_driving_voltage = 0.1 # driving voltage [V] for constant input current
const_c_Na_extracell = 145.   # mmol 
const_c_K_extracell =  5.  # mmol 
const_c_Cl_extracell = 110.  # mmol 
const_c_Na_rest = 10.  # intracell. sodium concentration at rest
const_c_K_rest = 140. # intracell. sodium concentration at rest
const_c_Cl_rest = 10.  # intracell. sodium concentration at rest   
# for ion concentrations see: https://bionumbers.hms.harvard.edu/files/Comparison%20of%20ion%20concentrations%20inside%20and%20outside%20a%20typical%20mammalian%20cell.jpg
# for membrane capacitance see: https://bionumbers.hms.harvard.edu/bionumber.aspx?&id:110759
# membrane resistance see dyan & abbott p. 207

scaling1 = {  
    # multiplicative scaling factors to make equations dimensionless
    'scale_voltage' : scale_voltage,
    'scale_space' : scale_space,
    'scale_time' : scale_time,
    'scale_concentration' : scale_concentration,
    'scale_diffusion' : scale_diffusion,
    'scale_charge' : scale_charge,
    'scale_current' : scale_current,
    'scale_resistance' : scale_resistance,
    'scale_capacitance' : scale_capacitance,
}

params1 = {  # standard parameter set
     # constants
    'const_k_B': const_k_B,
    'const_e' : const_e,
    'const_N_A' : const_N_A,  
    'const_z_Na' : const_z_Na,
    'const_z_K' : const_z_K,
    'const_z_Cl' : const_z_Cl, 
    'const_z_background' : const_z_background,

    # spine parameters, all set in SI-units
    'const_D_Na' : const_D_Na, 
    'const_D_K' : const_D_K,  
    'const_D_Cl' : const_D_Cl,  
    'const_r_m_Na' : const_r_m_Na, 
    'const_r_m_K' : const_r_m_K, 
    'const_r_m_Cl' : const_r_m_Cl, 
    'const_c_m' : const_c_m, 
    'const_T' : const_T,  
    'const_phi_rest' : const_phi_rest, 
    'const_c_Na_extracell' : const_c_Na_extracell,  
    'const_c_K_extracell' :  const_c_K_extracell, 
    'const_c_Cl_extracell' : const_c_Cl_extracell, 
    'const_c_Na_rest' : const_c_Na_rest, 
    'const_c_K_rest' : const_c_K_rest,  
    'const_c_Cl_rest' : const_c_Cl_rest, 
    'const_driving_voltage' : const_driving_voltage,
    }

params2 = copy.copy(params1)
params2['const_D_Na'] = const_D_K

parameter_sets = {# contains all parameter sets as dictionnaries
    'standard' : params1,
    'equal_diffusion': params2,
    }

scalings = {
    'standard': scaling1,
}
