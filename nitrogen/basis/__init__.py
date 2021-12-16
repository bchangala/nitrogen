"""
nitrogen.basis 
--------------

Basis set functions including discrete-variable 
representations (DVRs) and finite-basis representations
(FBRs). The main objects are the :class:`GriddedBasis` class
and its sub-classes :class:`GenericDVR` and :class:`NDBasis`.
See :doc:`tutorials/dvr` for a tutorial.

================================   ===================================
General gridded bases
======================================================================
:class:`GriddedBasis`              General quadrature grid basis.
:class:`ConcatenatedBasis`         Direct sum of :class:`GriddedBasis`
--------------------------------   -----------------------------------
**Discrete-variable representation bases**
----------------------------------------------------------------------
:class:`GenericDVR`                Parent DVR basis class.
:class:`SimpleDVR`                 Simple one-dimensional DVRs.
:class:`Contracted`                Contracted DVR.
--------------------------------   -----------------------------------
**FBR quadrature bases**
----------------------------------------------------------------------
:class:`NDBasis`                   Parent class for :math:`n`-d quadrature bases
:class:`SinCosBasis`               A sine-cosine (real Fourier) basis.
:class:`LegendreLMCosBasis`        Associated Legendre polynomials.
:class:`RealSphericalHBasis`       Real spherical harmonics.
:class:`Real2DHOBasis`             Two-dimensional harmonic oscillator.
:class:`RadialHOBasis`             Radial HO basis in :math:`d` dimensions.
================================   ===================================


"""

# Import main module into name-space
from . import genericbasis 
from .genericbasis import *

from . import dvr
from .dvr import *

from . import ndbasis 
from .ndbasis import *

# Load submodules
from . import ops  # DVR operators

__all__ = [] 
__all__ += genericbasis.__all__
__all__ += dvr.__all__
__all__ += ndbasis.__all__ 

import numpy as np 
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from skimage.measure import marching_cubes
from scipy import interpolate
    


def gridshape(dvrs):
    """
    Return the shape of the N-D formed by a list of DVRs.

    Parameters
    ----------
    dvrs : list
        Each element is a GenericDVR object or a fixed-value
        scalar.
        

    Returns
    -------
    shape : tuple
        The N-D DVR grid shape. 

    """
    
    shape = []
    
    for i in range(len(dvrs)):
        
        if isinstance(dvrs[i], dvr.GenericDVR): 
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
        Each element is a GenericDVR object or a fixed-value
        scalar.
        
    Returns
    -------
    grid : ndarray 
        An (N+1)-D ndarray of the stacked meshgrids

    """
    
    grids = []
    vshape = []
    
    for i in range(len(dvrs)):
        
        if isinstance(dvrs[i], dvr.GenericDVR): 
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

def bases2grid(bases):
    """
    Create direct product grids from a list of GriddedBasis objects
    and scalars.

    Parameters
    ----------
    bases : list
        Each element is a :class:`~nitrogen.genericbasis.GriddedBasis` object or a 
        fixed-value scalar.

    Returns
    -------
    grid : ndarray
        

    """
    
    grids = [] 
    qshape = []
    nq = 0
    
    index_of_coord = []
    
    for i,bas in enumerate(bases):
        
        # if isinstance(bas, GenericDVR):
        #     grids.append(bas.grid)
        #     qshape.append(bas.num)
        #     nq += 1
        #     index_of_coord.append(i)
        # elif isinstance(bas, NDBasis):
        #     for j in range(bas.nd) :
        #         grids.append(bas.qgrid[j])
        #         index_of_coord.append(i)
        #     qshape.append(bas.Nq)
        #     nq += bas.nd
        # else: 
        #     grids.append(bas)
        #     qshape.append(1) 
        #     nq += 1    
        #     index_of_coord.append(i)
        
        # Use generic GriddedBasis interface
        if np.isscalar(bas):
            grids.append(bas) # The scalar value
            qshape.append(1)
            nq += 1 
            index_of_coord.append(i)
        else:
            # Assume GriddedBasis
            for j in range(bas.nd):
                grids.append(bas.gridpts[j])
                index_of_coord.append(i)
            qshape.append(bas.ng)
            nq += bas.nd
            
    Qi = []
    for i in range(nq):
        # The i**th coordinate
        # spans the j**th index (axis)
        j = index_of_coord[i] 
        
        newshape = [1]*len(qshape)
        newshape[j] = qshape[j]
        newshape = tuple(newshape) 
        gi = np.reshape(grids[i], newshape) 
        
        Qi.append(np.broadcast_to(gi, qshape))
        
    Q = np.stack(Qi, axis = 0) 
    
    return Q
    

def plot(dvrs, fun, labels = None,
            ls = 'k.-', mode2d = 'surface', isovalue = None):
    """
    Plot a function over a 1-D or 2-D DVR grid

    Parameters
    ----------
    dvrs : list
        List of GenericDVRs or fixed-value scalars.
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
            try:
                verts, faces, _, _ = marching_cubes(V, iso_val[i], step_size = 1)
                vxyz = [interpolate.interp1d(np.arange(dvrs[idx[i]].num), dvrs[idx[i]].grid)(verts[:,i]) for i in range(3)]
                ax.plot_trisurf(vxyz[0], vxyz[1], faces, vxyz[2],
                                    lw=1, color = color[i])
            except:
                pass
                # marching_cubes will raise an error if
                # the surface isovalue is not in the range. 
                # If this occurs, we will just not plot that 
                # isosurfaces
                
                
            
            
        if labels is not None:
            ax.set_xlabel(labels[idx[0]])
            ax.set_ylabel(labels[idx[1]])
            ax.set_zlabel(labels[idx[2]])

    else:
        raise ValueError("There are more than 3 dimensions.")
        
    return fig, ax


def transferDVR(grid_old, dvrs_old, dvrs_new):
    """
    Interpolate a direct-product DVR expansion from one
    set of DVR bases to another.

    Parameters
    ----------
    grid_old : ndarray
        A grid of coefficients for the old DVR direct-product grid,
        with shape dvr2grid(dvrs_old).shape[1:]
    dvrs_old: list
        A list of GenericDVRs and/or scalar values. This is the original
        set of grids defining `grid_old`.
    dvrs_new : list
        A list of GenericDVRS and/or scalar values. This defines the 
        new direct product grid. Scalar elements in `dvrs_new` 
        must occur at the same position in `dvrs_old`, but their
        values are ignored.

    Returns
    -------
    grid_new : ndarray
        An array with shape dvr2grid(dvrs_new).shape[1:] containing
        the expansion coefficients for the function represented by
        `grid_old`.

    """
    
    grids_old = []
    vshape_old = []
    
    grids_new = []
    vshape_new = [] 
    
    nd = len(dvrs_old) # The number of dimensions, including singletons
    
    if nd != len(dvrs_new):
        raise ValueError("dvrs_old and dvrs_new must be the same length")
    
    for i in range(nd):
        
        if isinstance(dvrs_old[i], dvr.GenericDVR): 
            # Grid coordinate
            grids_old.append(dvrs_old[i].grid)
            vshape_old.append(dvrs_old[i].num)
            
            if not isinstance(dvrs_new[i], dvr.GenericDVR):
                raise TypeError("DVR vs. scalar mis-match")
            else:
                grids_new.append(dvrs_new[i].grid)
                vshape_new.append(dvrs_new[i].num)
            
        else:
            # Fixed coordinate
            grids_old.append(None) # None signifies non-active here
            vshape_old.append(1)
            grids_new.append(None) # None signifies non-active here
            vshape_new.append(1)
            
        
    # Grid shape, including singleton fixed coordinates
    vshape_old = tuple(vshape_old)
    vshape_new = tuple(vshape_new)
    
    # Evaluate the original expansion on the grid points
    # of the new expansion 
    
    eval_old_on_new = grid_old
    for i in range(nd):
        if grids_old[i] is None: 
            continue # This is a singleton/fixed dimension. Skip it
        # For the i**th dimension
        # 1) Evaluate the old DVR wavefunctions
        #    on the new DVR grid points
        # 
        ti = dvrs_old[i].wfs(dvrs_new[i].grid) # Shape: (num_new, num_old)
        #
        # 2) Convert the i**th dimension from old coefficients to the value
        #    on the new grid points
        eval_old_on_new = np.tensordot(ti, eval_old_on_new, axes = (1,i))
        eval_old_on_new = np.moveaxis(eval_old_on_new, 0, i)
    #
    # eval_old_on_new should now be complete

    # Calculate the weights of the new direct-product DVR on its own grid.
    wgt_new = np.ones(vshape_new)
    for i in range(nd):
        if grids_new[i] is not None:
            
            # Calculate the diagonal basis function values for this DVR
            wi = np.diag(dvrs_new[i].wfs(dvrs_new[i].grid))
            sh = [1 for i in range(len(grids_new))]
            sh[i] = dvrs_new[i].num
            sh = tuple(sh) # (1, 1, ... , num, ..., 1, 1)
            # Broadcast these weights to the ND grid 
            wgt_new *= wi.reshape(sh)
    
    # Finally, calculate the coefficients of the new DVR functions
    grid_new = eval_old_on_new / wgt_new 
    
    return grid_new 

def _to_quad(bases, x, force_copy = False):
    
    """ convert the mixed FBR representation
    array to the quadrature array
    """
    for i,b in enumerate(bases):
        # i**th axis
        if b is None or np.isscalar(b):
            pass 
        else:
            x = b.basis2grid(x,i)
    
    if force_copy:
        x = x.copy() 

    return x 

def _to_fbr(bases, x, force_copy = False):
    
    """ convert the quadrature array to 
    the mixed FBR representation array 
    """
    for i,b in enumerate(bases):
        # i**th axis
        if b is None or np.isscalar(b):
            pass 
        else:
            x = b.grid2basis(x,i)
    
    if force_copy:
        x = x.copy() 

    return x 

def calcRhoLogD(bases, Q):
    """
    Calculate the logarithmic derivative
    of the basis set volume integration
    function.

    Parameters
    ----------
    bases : list
        List of GriddedBasis and scalars.
    Q : ndarray
        The coordinate values. `Q[i]` is an array
        for the i**th coordinate.

    Returns
    -------
    rhotilde : ndarray
        `rhotilde[i]` is the logarithmic derivative
        of :math:`\\rho` with respect to the `i`th 
        **active** coordinate.

    """
    
    rhotilde = [] 
    k = 0
    for b in bases:
    
        if np.isscalar(b):
            # An inactive coordinate
            # No entry
            k += 1 
        else:
            # Assume a GriddedBasis
            if b.wgtfun is None:
                for i in range(b.nd):
                    rhotilde.append(np.zeros(Q.shape[1:]))
                    k += 1
            else:
                # GriddedBasis objects provide their weight
                # function with the DFun wgtfun()
                # Evaluate wgtfun and its first derivatives
                # over the quadrature grid. It only takes
                # the coordinates belonging to this basis 
                # set as arguments
                #
                # Note: using the *entire* quadrature grid is a big 
                # waste of effort because most of the arrays are the same value
                # One could slice-out the necessary coordinates and then
                # broadcast them back out to the entire quadrature grid,
                # but wgtfun is usually a simple, inexpensive function
                # so this is not a bottle-neck.
                #
                rho = b.wgtfun.f(Q[k:(k+b.nd)], deriv = 1) 
                for i in range(b.nd): # for each coordinate in the basis
                    rhoi = rho[i+1][0] / rho[0][0] # calculate log. deriv.
                    rhotilde.append(rhoi) 
                    k += 1
                    
    rhotilde = np.stack(rhotilde, axis = 0) # active only 
    
    return rhotilde
        