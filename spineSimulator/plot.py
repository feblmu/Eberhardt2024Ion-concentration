#/usr/bin/python

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import numpy as np
import dbm.dumb as dbm


def figure_main_axes_overview(file_name, times=[0.0005, 0.0095, 0.0105, 0.0195]):
    results = load_resuls(file_name)
    t = results['t']
    x = results['x']
    phi = results['phi']
    c_Na = results['c_Na']
    c_K = results['c_K']
    c_Cl = results['c_Cl']

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

def load_resuls(file_name):
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
    }
    
    f.close()
    
    return results


def figure_head_overview(file_name):
    results = load_resuls(file_name)
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
    
    results = load_resuls(file_name)
    t = results['t']
    phi = results['phi']
    c_Na = results['c_Na']
    c_K = results['c_K']
    c_Cl = results['c_Cl']
    
    c_total = c_Na - c_Cl + c_K
    
    ax.plot(t, c_total[:,3])

def surface(fig, pos, x_grid, t_grid, var):

    ax = fig.add_axes(pos, projection="3d")
    
    surf = ax.plot_surface(x_grid, t_grid, var, cmap=cm.summer,
                       linewidth=0, antialiased=False)
                       
    # Customize the z axis.
    #ax.set_zlim(-1.01, 1.01)
    #ax.zaxis.set_major_locator(LinearLocator(10))
    # A StrMethodFormatter is used automatically
    #ax.zaxis.set_major_formatter('{x:.02f}')

    # Add a color bar which maps values to colors.
    ## fig.colorbar(surf, shrink=0.5, aspect=5)
    
    ax.set_xlabel('x')
    ax.set_ylabel('t')
    
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
    ax.set_xlabel('length [$nm$]')
    ax.set_yticklabels([int(round(yi,1)) for yi in ax.get_yticks() * 1.e9])
    ax.set_ylabel('radius [$nm$]')
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
