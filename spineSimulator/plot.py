#/usr/bin/python

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import numpy as np


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
