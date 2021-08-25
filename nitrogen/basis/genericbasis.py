# -*- coding: utf-8 -*-
"""
genericbasis.py
"""

import numpy as np 

__all__ = ["GriddedBasis", "ConcatenatedBasis",
           "basisShape", "coordAxis", "coordSubcoord", "basisVar", "sameGrid"]

class GriddedBasis:
    """
    A generic super-class for gridded basis sets,
    including DVRs and FBRs equipped with quadrature grids.
    
    Attributes
    ----------
    nb : int 
        The number of basis functions
    nd : int 
        The number of grid coordinates
    ng : int 
        The number of grid points 
    gridpts : (nd,ng) ndarray
        The grid points
    wgtfun : DFun
        An integration weight function. If None, this is unity.
    
    """
    
    def __init__(self, gridpts, nb, wgtfun = None):
        """
        Create a GriddedBasis

        Parameters
        ----------
        gridpts : (nd, ng) array_like
            The `ng` grid points for each of the `nd` coordinates.
        nb : int
            The number of basis functions.
        wgtfun : DFun, optional
            The integration weight function 

        """
        
        gridpts = np.array(gridpts)
        nd,ng = gridpts.shape 
        
        self.gridpts = gridpts   # The grid
        self.nb = nb       # The number of basis functions
        self.nd = nd       # The number of dimensions
        self.ng = ng       # The number of grid points 
        self.wgtfun = wgtfun 
    
    def basis2grid(self, x, axis = 0):
        return self._basis2grid(x, axis)
    def grid2basis(self, x, axis = 0):
        return self._grid2basis(x, axis)
    def basis2grid_d(self, x, var, axis = 0):
        if var < 0 or var > self.nd:
            raise ValueError("var is out of range")
        return self._basis2grid_d(x, var,axis)
    def grid2basis_d(self, x, var, axis = 0):
        if var < 0 or var > self.nd:
            raise ValueError("var is out of range")
        return self._grid2basis_d(x, var,axis)
    def d_grid(self, x, var, axis = 0):
        if var < 0 or var > self.nd:
            raise ValueError("var is out of range")
        return self._d_grid(x, var,axis)
    def dH_grid(self, x,  var,axis = 0):
        if var < 0 or var > self.nd:
            raise ValueError("var is out of range")
        return self._dH_grid(x, var,axis)
    
    def _basis2grid(self, x, axis = 0):
        raise NotImplementedError()
    def _grid2basis(self,x, axis = 0):
        raise NotImplementedError()
    def _basis2grid_d(self,x, var, axis = 0):
        raise NotImplementedError()
    def _grid2basis_d(self, x, var, axis = 0):
        raise NotImplementedError()
    def _d_grid(self,x, var, axis = 0):
        raise NotImplementedError()
    def _dH_grid(self,x, var, axis = 0):
        raise NotImplementedError()

class ConcatenatedBasis(GriddedBasis):
    """
    A concatenated set of GriddedBasis basis sets.
    It is assumed these have compatible grid methods.
    
    """
    
    def __init__(self, bases):
        
        nbases = len(bases)
        if nbases < 1:
            raise ValueError("There must be at least 1 basis set.")
        for i in range(nbases-1):
            if not sameGrid(bases[i], bases[i+1]):
                raise ValueError("Basis sets appear incompatible.")
                
        # Calculate the total number of basis functions 
        # and the look-up indices for each sub-set
        nb = 0
        bidx = []
        for b in bases:
            #bidx.append(np.arange(nb, nb + b.nb))
            bidx.append((nb, nb+b.nb)) # slice start/stop values 
            nb += b.nb
        
        super().__init__(bases[0].gridpts, nb, bases[0].wgtfun) 
        
        self.__bases = bases 
        self.__nbases = nbases 
        self.__bidx = bidx 
        
        return 
    
    def _basis2grid(self, x, axis = 0):
        y = 0
        #
        # Add each block of basis functions 
        # onto the same grid
        #
        for i in range(self.__nbases):
            indices = self.__bidx[i]
            b = self.__bases[i]
            idx = _create_nd_slice(x.ndim, axis, indices[0], indices[1])
            xi = x[idx]
            y += b.basis2grid(xi, axis = axis)
        return y 
    
    
    def _grid2basis(self, x, axis = 0):
        
        # Calculate the overlap with each block of 
        # basis functions, then concatenate along the axis
        #
        y_sub = [b.grid2basis(x, axis = axis) for b in self.__bases]
        return np.concatenate(y_sub, axis = axis)
    
    
    def _basis2grid_d(self, x, var, axis = 0):
        y = 0
        #
        # Add each block of basis functions 
        # onto the same grid
        #
        for i in range(self.__nbases):
            indices = self.__bidx[i]
            b = self.__bases[i]
            idx = _create_nd_slice(x.ndim, axis, indices[0], indices[1])
            xi = x[idx]
            y += b.basis2grid_d(xi, var, axis = axis)
        return y 
    
    def _grid2basis_d(self, x, var, axis = 0):
        
        # Calculate the overlap with each block of 
        # basis functions, then concatenate along the axis
        #
        y_sub = [b.grid2basis_d(x, var, axis = axis) for b in self.__bases]
        return np.concatenate(y_sub, axis = axis) 
    
    #
    # Leaving _d_grid and _dH_grid not implemented
    # for now. These methods don't quite make sense
    # for a ConcatenatedBasis, for which we cannot
    # guarantee quasi-unitarity.
    #
        
        

def basisShape(bases):
    """
    Determine the direct-product shape of a set of GriddedBasis
    and/or scalars.

    Parameters
    ----------
    bases : list of GriddedBasis and scalar
        The direct-product factors. Scalars become singleton dimensions.

    Returns
    -------
    shape : tuple 
        The basis shape.
    size : int
        The total size.

    """
    
    shape = []
    size = 1 
    for b in bases:
        if np.isscalar(b):
            shape.append(1)  # singleton dimension
        else:
            shape.append(b.nb) # the basis dimension
            size = size * b.nb 
    
    return tuple(shape), size 

def coordAxis(bases):
    """
    Determine the direct-product axis to which each coordinate belongs
    
    Parameters
    ----------
    bases : list of GriddedBasis and scalar
        The direct-product factors. Scalars become singleton dimensions.
        
    Returns
    -------
    axis_of_coord : list
        The direct-product axis of each coordinate.
    """
    
    axis_of_coord = []
    for ax,b in enumerate(bases):
        if np.isscalar(b):
            axis_of_coord.append(ax) # Singleton scalar
        else:
            for i in range(b.nd): # For each coordinate represented by this factor
                axis_of_coord.append(ax)
    
    return axis_of_coord

def coordSubcoord(bases):
    """
    Determine the intra-basis coordinate index of each coordinate represented
    by the complete basis. Scalar singletons will just take 0.
    
    Parameters
    ----------
    bases : list of GriddedBasis and scalar
        The direct-product factors. Scalars become singleton dimensions.
        
    Returns
    -------
    axis_of_coord : list
        The intra-basis coordinate sub-index of each coordinate.
    """
    subcoord_of_coord = []
    for b in bases:
        if np.isscalar(b):
            subcoord_of_coord.append(0) # Singleton scalar
        else:
            for i in range(b.nd):
                subcoord_of_coord.append(i)
    return subcoord_of_coord

def basisVar(bases):
    """
    Return the list of active coordinates (i.e. non-scalar entries) represented
    by a list of GriddedBasis or scalars
    
    Parameters
    ----------
    bases : list of GriddedBasis and scalar
        The direct-product factors. Scalars become singleton dimensions.
        
    Returns
    -------
    var : list
        The ordered list of active coordinates.

    """
          
    var = []  # The ordered list of active coordinates 
    k = 0
    for b in bases:
        if np.isscalar(b):
            # A singleton, inactive coordinate
            k += 1
        else:
            # An active basis
            for i in range(b.nd):
                var.append(k)
                k += 1 
    return var
    
def sameGrid(A, B):
    """
    Check whether GriddedBases A and B have the same grid.
    This also handles scalar input, in which case simple equality is checked.
    
    A, B : GriddedBasis or scalar
    
    Returns
    -------
    bool
        True if equal grids, otherwise False.
    """
    if np.isscalar(A):
        # A is a scalar 
        if not np.isscalar(B):
            return False
        else:
            return (A == B)
    else:
        # Assume two GriddedBasis objects 
        if A.nd != B.nd or A.ng != B.ng:
            return False 
        elif not np.all(A.gridpts == B.gridpts):
            return False 
        else:
            return True 
        
        
def _create_nd_slice(nd, axis, start, stop, step = None):
    
    pre = [slice(None)] * axis 
    idx = [slice(start, stop, step)] 
    post = [slice(None)] * (nd - axis - 1) 
    
    return tuple(pre + idx + post)
        
        
        
        