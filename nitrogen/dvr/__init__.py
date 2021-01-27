"""
nitrogen.dvr
------------

This module provides support for discrete-variable 
representation (DVR) basis functions. The main object
is the :class:`DVR` class.

"""

# Import main module into name-space
from . import dvr
from .dvr import *

# Load submodules
from . import ops  # DVR operators

__all__ = dvr.__all__

import numpy as np 
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm

def gridshape(dvrs):
    """
    Return the shape of the N-D formed by a list of DVRs.

    Parameters
    ----------
    dvrs : list
        Each element is a DVR object or a fixed-value
        scalar.
        

    Returns
    -------
    shape : tuple
        The N-D DVR grid shape. 

    """
    
    shape = []
    
    for i in range(len(dvrs)):
        
        if isinstance(dvrs[i], dvr.DVR): 
            # Grid coordinate
            shape.append(dvrs[i].num)
        else:
            # Fixed coordinate
            shape.append(1)
        
    # Grid shape, including singleton fixed coordinates
    shape = tuple(shape)
    
    return shape


def dvr2grid(dvrs):
    """
    Create N-D grids from a list of DVRs.

    Parameters
    ----------
    dvrs : list
        Each element is a DVR object or a fixed-value
        scalar.
        
    Returns
    -------
    grid : ndarray 
        An (N+1)-D ndarray of the stacked meshgrids

    """
    
    grids = []
    vshape = []
    
    for i in range(len(dvrs)):
        
        if isinstance(dvrs[i], dvr.DVR): 
            # Grid coordinate
            grids.append(dvrs[i].grid)
            vshape.append(dvrs[i].num)
        else:
            # Fixed coordinate
            grids.append(dvrs[i]) # scalar value
            vshape.append(1)
        
    # Grid shape, including singleton fixed coordinates
    vshape = tuple(vshape)

    # Calculate the coordinate grids
    Q = np.stack(np.meshgrid(*grids, indexing = 'ij'))
    
    return Q

def plot(dvrs, fun, labels = None,
            ls = 'k.-', mode2d = 'surface'):
    """
    Plot a function over a 1-D or 2-D DVR grid

    Parameters
    ----------
    dvrs : list
        List of DVRs or fixed-value scalars.
    fun : function or array
        If function, f(Q) evaluates the vectorized grid function.
        If an array, then fun is the same size of the return of
        dvr2grid(dvrs).shape[1:]
    labels : list of str, optional
        Coordinate labels (including fixed).
    ls : str, optional
        1-D line spec.
    mode2d: {'surface', 'contour'}, optional
        2-d plot style. 

    Returns
    -------
    fig, ax
        Plot objects

    """
    
    qgrid = dvr2grid(dvrs)
    try: # Attempt to call fun as a function
        ygrid = fun(qgrid)
    except:
        # If that fails, assume it is an ndarray grid
        ygrid = fun.copy()
    
    # Determine the non-singleton dimensions
    idx = []
    for i in range(len(ygrid.shape)):
        if ygrid.shape[i] > 1:
            idx.append(i)
            
    ndim = len(idx)
    
    if ndim < 1:
        raise ValueError("There must be at least 1 non-singleton dimension")
    
    elif ndim == 1:
        #
        # 1-D plot
        #
        fig = plt.figure()
        
        x = qgrid[idx[0]].squeeze() 
        y = ygrid.squeeze()
        plt.plot(x,y,ls)
        ax = plt.gca()
        
        if labels is not None:
            plt.xlabel(labels[idx[0]])
    
    elif ndim == 2:
        #
        # 2-D surface or contour plot 
        #
        
        fig = plt.figure()
        
        X = qgrid[idx[0]].squeeze() 
        Y = qgrid[idx[1]].squeeze()
        Z = ygrid.squeeze() 
            
        if mode2d == 'surface':
            
            ax = fig.add_subplot(111, projection='3d')
            ax.plot_surface(X, Y, Z,
                           cmap = cm.coolwarm,
                           linewidth = 0,
                           rcount = Z.shape[0],
                           ccount = Z.shape[1])
        elif mode2d == 'contour':
            
            plt.contour(X, Y, Z, levels = 50) 
            ax = plt.gca() 
            
        else:
            raise ValueError("Unexpected mode2d string")
            
        if labels is not None: 
            ax.set_xlabel(labels[idx[0]])
            ax.set_ylabel(labels[idx[1]])
        
    else:
        raise ValueError("There are more than 2 dimensions.")
        
    return fig, ax