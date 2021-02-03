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
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from skimage.measure import marching_cubes


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
            ls = 'k.-', mode2d = 'surface', isovalue = None):
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
        2-D plot style. 
    isovalue : scalar or array_like
        Isosurface value(s) for 3-D plot. If None (default), a 
        fixed fraction of the maximum absolute value will be used.

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
        
    elif ndim == 3:
        # 
        # 3-D isosurface plot
        #
        #x, y, z = pi*np.mgrid[-1:1:31j, -1:1:31j, -1:1:31j]
        #vol = cos(x) + cos(y) + cos(z)
        X = qgrid[idx[0]].squeeze() 
        Y = qgrid[idx[1]].squeeze()
        Z = qgrid[idx[2]].squeeze()
        V = ygrid.squeeze() # Value field
        
        
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        
        if isovalue is None:
            iso_val = np.array([ i * 0.2 * np.max(np.absolute(V)) for i in [-1., 1.]])
        elif np.isscalar(isovalue):
            iso_val = np.array([isovalue])
        else:
            iso_val = np.array(isovalue)


        if len(iso_val) == 0:
            raise ValueError("There must be at least one isovalue.")
        elif len(iso_val) == 1:
            color = ['b']
        else:
            min_val = np.min(iso_val)
            max_val = np.max(iso_val)
            P = (iso_val - min_val) / (max_val - min_val) 
            color = [ (1.0-p, 0.0, p) for p in P]
            
        for i in range(len(iso_val)):
            # Positive iso value
            verts, faces, _, _ = marching_cubes(V, iso_val[i], spacing=(0.1, 0.1, 0.1))
            ax.plot_trisurf(verts[:, 0], verts[:,1], faces, verts[:, 2],
                            lw=1, color = color[i])
        
        if labels is not None:
            ax.set_xlabel(labels[idx[0]])
            ax.set_ylabel(labels[idx[1]])
            ax.set_zlabel(labels[idx[2]])

    else:
        raise ValueError("There are more than 3 dimensions.")
        
    return fig, ax