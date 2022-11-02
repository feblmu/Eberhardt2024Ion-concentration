#/usr/bin/python

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import numpy as np
import dbm.dumb as dbm

#########################################

# FILE

#########################################

def load_results(file_name):
    f = dbm.open('./../../simulation_results/' + file_name, 'r')
    t = np.frombuffer(f['t'])
    results = {
        't': t,
        'x': np.frombuffer(f['x']),
        'a': np.frombuffer(f['radius']),
        'phi': np.array([np.frombuffer(f['phi'+str(ti)]) for ti in range(len(t))]),
        'c_Na': np.array([np.frombuffer(f['c_Na'+str(ti)]) for ti in range(len(t))]),
        'c_K': np.array([np.frombuffer(f['c_K'+str(ti)]) for ti in range(len(t))]),
        'c_Cl': np.array([np.frombuffer(f['c_Cl'+str(ti)]) for ti in range(len(t))]),
        'parameters': f[b'parameters']
    }
    
    f.close()
    
    return results

def get_results_summary(file_id):
    """
    load main results of a partiuclar spine setting
    and compute relevant parameters for analysis
    """
    # load data from files
    from constants import parameter_sets
    results = load_results(file_id)
    
    x = results['x']
    t = results['t']
    a = results['a']
    phi = results['phi']
    c_Na = results['c_Na']
    c_K = results['c_K']
    c_Cl = results['c_Cl']

    param_set = results['parameters'].decode("utf-8") # database contains a bytestring which needs to be converted to str, e.g. 'standard'
    params = parameter_sets[param_set]

    dx = x[1] - x[0]
    dt = t[1] - t[0]

    #####################################
    r_Na = compute_resistivity(c_Na, params, 'Na')
    R_Na = compute_resistance(r_Na, a, dx)
    g_Na_ij = compute_conductivity(r_Na, a, dx)

    r_K = compute_resistivity(c_K, params, 'K')
    R_K = compute_resistance(r_K, a, dx)
    g_K_ij = compute_conductivity(r_K, a, dx)

    r_Cl = compute_resistivity(c_Cl, params, 'Cl')
    R_Cl = compute_resistance(r_Cl, a, dx)
    g_Cl_ij = compute_conductivity(r_Cl, a, dx)

    r_e = 1./ ( 1./r_Na + 1./r_K + 1./r_Cl )
    R_e = compute_resistance(r_e, a, dx)
    g_ij = compute_conductivity(r_e, a, dx)

    i_c_Na = compute_chemical_current(c_Na, a, dx, params, 'Na')
    i_c_K = compute_chemical_current(c_K, a, dx, params, 'K')
    i_c_Cl = compute_chemical_current(c_Cl, a, dx, params, 'Cl')

    i_e_Na = compute_electrical_current(g_Na_ij, phi)
    i_e_K = compute_electrical_current(g_K_ij, phi)
    i_e_Cl = compute_electrical_current(g_Cl_ij, phi)

    # chemical potentials
    #mu_Na = compute_chemical_potential(c_Na, params)
    #mu_K = compute_chemical_potential(c_K, params)
    #mu_Cl = compute_chemical_potential(c_Cl, params)

    i_e = i_e_Na + i_e_K + i_e_Cl

    i_c = i_c_Na + i_c_K + i_c_Cl

    i_total = i_e + i_c
    
    return (
        x,
        a,
        t,
        phi,
        c_Na,
        c_K,
        c_Cl,
        param_set,
        params,
        dx,
        dt,
        r_Na,
        R_Na,
        g_Na_ij,
        r_K,
        R_K,
        g_K_ij,
        r_Cl,
        R_Cl,
        g_Cl_ij,
        r_e,
        R_e,
        g_ij,
        i_c_Na,
        i_c_K,
        i_c_Cl,
        i_e_Na,
        i_e_K,
        i_e_Cl,
        i_e,
        i_c,
        i_total,
    )

#########################################

# ANALYSIS

#########################################

def compute_resistivity(c, params, ion=''):
    """
    ion: can be "Na", "K" or "Cl"
    """
    V_T = params['const_k_B'] * params['const_T'] / params['const_e']
    f = params['const_D_'+ion] * params['const_e'] * params['const_z_'+ion]**2 * c * params['const_N_A']
    r = V_T / f
    return r

def compute_resistance(r, a, dx):
    R = r * dx / np.pi / np.square(a)
    return R
    
def compute_conductivity(r, a, dx):
    #g_ij = (np.square(a[1:]) / r[:,1:] + np.square(a[:-1]) / r[:,:-1]) * np.pi / 2. / dx
    g_ij = 2. * (np.square(a[1:]) * np.square(a[:-1])) / (r[:,1:]*np.square(a[1:])+r[:,:-1]*np.square(a[:-1])) / dx * np.pi
    return g_ij
    
def compute_chemical_current(c, a, dx, params, ion):
    #i_c = (- params['const_D_'+ion] * params['const_e'] * params['const_z_'+ion] * np.pi* 
    #      (np.square(a[1:]) + np.square(a[:-1])) /2. * 
    #      params['const_N_A']*(c[:, 1:] - c[:,:-1])/ dx)
    i_c = (- params['const_D_'+ion] * params['const_e'] * params['const_z_'+ion] * np.pi* 
          2.* (np.square(a[1:]) * np.square(a[:-1])) / (np.square(a[1:]) + np.square(a[:-1]))  * 
          params['const_N_A']*(c[:, 1:] - c[:,:-1])/ dx)
    return i_c

def compute_electrical_current(g_ij, phi):
    i_e = - g_ij * (phi[:, 1:] - phi[:,:-1])
    return i_e
    
#def compute_chemical_potential(c, params):
#    mu = params['const_k_B'] * params['const_T'] * np.log(c / params['const_c_Na_rest'])
#    return mu


#########################################

# VISUALIZATION

#########################################

def ax_surface(fig, pos, x_grid, t_grid, var, z_label='', layout='standard'):

    ax = fig.add_axes(pos, projection="3d")
    
    scale_space = 1.e9
    scale_time = 1.e3
    if layout == 'Phi':
        scale_var = 1.e3
    else: scale_var = 1.
    
    surf = ax.plot_surface(x_grid*scale_space, t_grid*scale_time, var*scale_var, cmap=cm.summer,
                       linewidth=0, antialiased=False)
                       
    # Customize the z axis.
    #ax.set_zlim(-1.01, 1.01)
    #ax.zaxis.set_major_locator(LinearLocator(10))
    # A StrMethodFormatter is used automatically
    #ax.zaxis.set_major_formatter('{x:.02f}')

    # Add a color bar which maps values to colors.
    ## fig.colorbar(surf, shrink=0.5, aspect=5)
    
    ax.set_xlabel('z [nm]', labelpad=-10)
    ax.set_ylabel('t [ms]', labelpad=-10)
    ax.set_zlabel(z_label, rotation=90, labelpad=0)
    
    ax.tick_params(axis='x', which='major', pad=-5)
    ax.tick_params(axis='y', which='major', pad=-5)
    ax.tick_params(axis='z', which='major', pad=0)

def figure_space_time_summary(file_name):
    results = load_results(file_name)
    t = results['t']
    x = results['x']
    phi = results['phi']
    c_Na = results['c_Na']
    c_K = results['c_K']
    c_Cl = results['c_Cl']
    
    xx, tt = np.meshgrid(x,t)

    fig=plt.figure(dpi=300, facecolor='white', figsize=(5,4))
    #fig.suptitle(file_name)

    wx, wy = 0.35, 0.35
    pos1 = [0.15, 0.55,wx, wy]
    pos2 = [0.6, 0.55, wx, wy]
    pos3 = [0.15, 0.1, wx, wy]
    pos4 = [0.6, 0.1, wx, wy]
    
    ax_surface(fig, pos1, xx, tt, phi, z_label=r'$\Phi$ [mV]', layout='Phi')
    ax_surface(fig, pos2, xx, tt, c_Na, z_label=r'$c_{Na}$')
    ax_surface(fig, pos3, xx, tt, c_K, z_label=r'$c_K$ [mmol]')
    ax_surface(fig, pos4, xx, tt, c_Cl, z_label=r'$c_{Cl}$ [mmol]')

def figure_main_axes_overview(file_name, times=[0.0005, 0.0095, 0.0105, 0.0195]):
    results = load_results(file_name)
    t = results['t']
    x = results['x']
    phi = results['phi']
    c_Na = results['c_Na']
    c_K = results['c_K']
    c_Cl = results['c_Cl']

    fig=plt.figure(dpi=300, facecolor='white', figsize=(5,4))
    fig.suptitle(file_name)

    wx, wy = 0.3, 0.3
    pos1 = [0.15, 0.6,wx, wy]
    pos2 = [0.6, 0.6, wx, wy]
    pos3 = [0.15, 0.1, wx, wy]
    pos4 = [0.6, 0.1, wx, wy]

    ax1 = fig.add_axes(pos1)
    ax2 = fig.add_axes(pos2)
    ax3 = fig.add_axes(pos3)
    ax4 = fig.add_axes(pos4)
    
    for t_ in times:
        t_i = np.sum(t<=t_)
        ax1.plot(x, phi[t_i,:], label=str(t_))
        ax2.plot(x, c_Na[t_i,:])
        ax3.plot(x, c_K[t_i,:])
        ax4.plot(x, c_Cl[t_i,:])

    y_lables = [
        'Potential [V]',
        'Sodium\n Concentration [mmol]',
        'Potassium\n Concentration [mmol]',
        'Chloride\n Concentration [mmol]',
    ]

    for i, ax in enumerate([ax1, ax2, ax3, ax4]):
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xlabel('Distance from Synapse [m]', fontsize=8)
        ax.set_ylabel(y_lables[i], fontsize=8)
        ax.tick_params(labelsize=8)
    
    ax1.legend(fontsize=8, loc=(.6,.6))
    plt.show()

def figure_head_overview(file_name):
    results = load_results(file_name)
    t = results['t']
    phi = results['phi']
    c_Na = results['c_Na']
    c_K = results['c_K']
    c_Cl = results['c_Cl']
    
    print(np.shape(phi))
    
    fig=plt.figure(dpi=300, facecolor='white', figsize=(5,4))
    fig.suptitle(file_name)

    wx, wy = 0.3, 0.3
    pos1 = [0.1, 0.6,wx, wy]
    pos2 = [0.6, 0.6, wx, wy]
    pos3 = [0.1, 0.1, wx, wy]
    pos4 = [0.6, 0.1, wx, wy]

    ax1 = fig.add_axes(pos1)
    ax2 = fig.add_axes(pos2)
    ax3 = fig.add_axes(pos3)
    ax4 = fig.add_axes(pos4)

    ax1.plot(t, phi[:,1])
    ax2.plot(t, c_Na[:,1])
    ax3.plot(t, c_K[:,1])
    ax4.plot(t, c_Cl[:,1])

    y_lables = [
        'Potential [V]',
        'Sodium\n Concentration [mmol]',
        'Potassium\n Concentration [mmol]',
        'Chloride\n Concentration [mmol]',
    ]

    for i, ax in enumerate([ax1, ax2, ax3, ax4]):
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xlabel('time [s]', fontsize=8)
        ax.set_ylabel(y_lables[i], fontsize=8)
        ax.tick_params(labelsize=8)
    
    plt.show()

def ax_electroneutrality_head(fig, pos, file_name):
    ax = fig.add_axes(pos)
    
    results = load_results(file_name)
    t = results['t']
    phi = results['phi']
    c_Na = results['c_Na']
    c_K = results['c_K']
    c_Cl = results['c_Cl']
    
    c_total = c_Na - c_Cl + c_K
    
    ax.plot(t, c_total[:,3])
    
def ax_electroneutrality_main_axis(fig, pos, file_name, t_i):
    ax = fig.add_axes(pos)
    
    results = load_results(file_name)
    x = results['x']
    t = results['t']
    phi = results['phi']
    c_Na = results['c_Na']
    c_K = results['c_K']
    c_Cl = results['c_Cl']
    
    c_total = c_Na - c_Cl + c_K
    
    ax.plot(x, c_total[t_i,:])
    
def x_grid_on_spine(fig, pos, x, a):
    
    ax = fig.add_axes(pos)
    
    delta_x = x[1] - x[0]
    
    for xi, ai in zip(x,a):
        x_tmp = np.array([xi - delta_x/2., xi + delta_x/2.])
        a_tmp = np.array([ai, ai])
        ax.fill_between(x_tmp, a_tmp, -a_tmp, color='lightgray', edgecolor='gray', lw=.5)
        ax.plot([xi],[0], 'ko', ms=2.,)
    #ax.plot(x,np.zeros(len(x)), 'k-', lw=.5)
    
    # scale axsis
    ax.set_xticklabels([int(round(xi,1)) for xi in ax.get_xticks() * 1.e9])
    ax.set_xlabel('z - main axis [$nm$]')
    ax.set_yticklabels([int(round(yi,1)) for yi in ax.get_yticks() * 1.e9])
    ax.set_ylabel('a - radius [$nm$]')
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    
def ax_extended_cable(fig, pos):
    ax = fig.add_axes(pos)

    ax.set_xticks([])
    ax.set_yticks([])

    l = 0.6
    h = 0.4

    ax.fill_between([-l/2., l/2.],[-h/2.,-h/2.], [h/2., h/2.], facecolor='lightgrey', edgecolor='gray')
    ax.plot([-l/2., l/2.],[-h/2.,-h/2.], 'k-', lw=2.)
    ax.plot([-l/2., l/2.],[h/2.,h/2.], 'k-', lw=2.)


    ax.annotate(r"$-\frac{\pi a^2}{r_c}\frac{\partial \mu_k}{\partial x }|_{left}$", fontsize=8,
                xy=(-l/2*0.7, -h/2*0.7), xytext=(-l*0.8, -h/2.*0.7), arrowprops=dict(arrowstyle="->"), ha='center', va='center')

    ax.annotate(r"$-\frac{\pi a^2}{r_e}\frac{\partial \Phi}{\partial x }|_{left}$", 
                xy=(-l/2*0.7, h/2*0.7), xytext=(-l*0.8, h/2.*0.7), arrowprops=dict(arrowstyle="->"), ha='center', va='center')

    ax.annotate(r"$-\frac{\pi a^2}{r_c}\frac{\partial \Phi}{\partial x }|_{right}$", 
                xy=(l/2*0.7, h/2*0.7), xytext=(l*0.8, h/2.*0.7), arrowprops=dict(arrowstyle="<-"), ha='center', va='center')

    ax.annotate(r"$-\frac{\pi a^2}{r_c}\frac{\partial \mu_k}{\partial x }|_{right}$", 
                xy=(l/2*0.7, -h/2*0.7), xytext=(l*0.8, -h/2.*0.7), arrowprops=dict(arrowstyle="<-"), ha='center', va='center')

    ax.annotate(r"$2 \pi a \Delta x c_m \frac{\partial \Phi}{\partial t }$", 
                xy=(0., -h/2.*0.7), xytext=(0., -0.8*h), arrowprops=dict(arrowstyle="->"), ha='center', va='center')

    #ax.annotate(r"$2 \pi a \Delta x ~ i_{AMPA}$",
    #            xy=(0., h/2.*0.7), xytext=(0., 0.8*h), arrowprops=dict(arrowstyle="->"), ha='center', va='center')

    ax.annotate(r"$z - main~axis$",
                xy=(-0.3, 0.8*h), xytext=(0., 0.8*h), arrowprops=dict(arrowstyle="<-"), ha='center', va='center')

    ax.text(0., 0.05, '$\Phi, \mu_k$', va='center', ha='center', fontsize=8,)
    ax.text(0., -0.05, '$k \in \{{Na^+},{K^+},{Cl^{-}}\}$', va='center', ha='center', fontsize=8,)


    ax.set_xlim([-l,l])
    ax.set_ylim([-h,h])
