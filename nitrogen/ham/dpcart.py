# -*- coding: utf-8 -*-
"""
Simple direct-product-DVR Cartesian Hamiltonians
"""

__all__ = ['DirProdDvrCartN', 'DirProdDvrCartNQD']

import numpy as np 
from scipy.sparse.linalg import LinearOperator
import nitrogen.constants
import nitrogen.basis.ops as dvrops 

class DirProdDvrCartN(LinearOperator):
    """
    A LinearOperator subclass for direct-product DVR
    Hamiltonians for N Cartesian coordinates.
    
    
    Attributes
    ----------
    V : ndarray
        The PES grid with shape `vshape`.
    NH : int
        The size of the Hamiltonian (the number of direct-product grid points).
    vvar : list 
        The active coordinates.
    grids : list
        Grids (for active) and fixed scalar values (for inactive) 
        of each coordinate.
    vshape : tuple
        The shape of the direct-product grid.
    masses : ndarray
        The mass of each coordinate.
    hbar : float
        The value of :math:`\\hbar`. 
    """
    
    def __init__(self, dvrs, pes, masses = None, hbar = None, Vmax = None, Vmin = None):
        """
        

        Parameters
        ----------
        dvrs : list of GenericDVR objects and/or scalars
            A list of :class:`nitrogen.basis.GenericDVR` objects and/or
            scalar numbers. The length of the list must be equal to
            the number of coordinates, `nx`. Scalar elements indicate
            a fixed value for that coordinate.
        pes : DFun or function
            A potential energy function f(X) with respect
            to the `nx` Cartesian coordinates.
        masses : array_like, optional
            A list of `nx` masses. If None, these will be assumed to be unity.
        hbar : float, optional
            The value of :math:`\\hbar`. If None, the default NITROGEN units are used.
        Vmax : float, optional
            Maximum potential energy allowed. Higher values will be 
            replaced with `Vmax`. If None, this is ignored.
        Vmin : float, optional
            Minimum potential energy allowed. Lower values will be 
            replaced with `Vmin`. If None, this is ignored.
            
        """
        
        nx = len(dvrs)
        
        if masses is None:
            masses = [1.0 for i in range(nx)]
        
        if len(masses) != nx:
            raise ValueError("The number of masses must equal the length of dvrs")

        if hbar is None:
            hbar = nitrogen.constants.hbar 
            
        # Determine the active and fixed coordinates
        vvar = []
        grids = []
        vshape = []
        NH = 1
        D2list = []
        
        for i in range(len(dvrs)):
            
            if np.isscalar(dvrs[i]): # inactive
                grids.append(dvrs[i])
                vshape.append(1)
                D2list.append(None) 
            else: # active, assume GenericDVR object
                vvar.append(i)
                grids.append(dvrs[i].grid)
                ni = dvrs[i].num # length of this dimension 
                vshape.append(ni) 
                NH *= ni 
                D2list.append(dvrs[i].D2)

        vshape = tuple(vshape)
        # To summarize:
        # -------------
        # vvar is a list of active coordinates
        # grids contains 1D active grids and fixed scalar values
        # vshape is the grid shape *including* singleton fixed coordinates
        # NH is the total grid size
        # D2list is the second-derivative operator list, including None's for fixed coord.
            
        if len(vvar) < 1:
            raise ValueError("There must be at least one active coordinate")
            
        # Calculate the coordinate grids
        Q = np.stack(np.meshgrid(*grids, indexing = 'ij'))
        
        # Calculate the PES grid
        try: # Attempt DFun interface
            V = pes.f(Q, deriv = 0)[0,0]
        except:
            V = pes(Q) # Attempt raw function
        # Check max and min limits
        if Vmax is not None:
            V[V > Vmax] = Vmax 
        if Vmin is not None: 
            V[V < Vmin] = Vmin 
        
        # Determine the operator data-type
        dtype = np.result_type(Q,V)
         
        # Initialize the LinearOperator
        #super().__init__((NH,NH), matvec = self._cartn_mv, dtype = dtype)
        # Define the required LinearOperator attributes
        self.shape = (NH,NH)
        self.dtype = dtype
        
        # Define new attributes
        self.NH = NH
        self.vvar = vvar 
        self.grids = grids 
        self.vshape = vshape 
        self._D2list = D2list 
        self.masses = np.array(masses)
        self.hbar = hbar 
        self.V = V
    
    def _matvec(self, x):
        
        xgrid = x.reshape(self.vshape) # Reshape vector to direct-product grid
        y = np.zeros_like(xgrid)        # Result grid
        
        hbar = self.hbar 
        ############################################
        # PES term
        y += self.V * xgrid 
        # KEO terms
        y += (-hbar**2 /2.0) * dvrops.opO_coeff(xgrid, self._D2list, 1.0/self.masses)
        #
        ############################################
                    
        # Reshape result grid to a 1D vector
        return y.reshape((self.NH,))        
    
class DirProdDvrCartNQD(LinearOperator):
    """
    A LinearOperator subclass for direct-product DVR
    Hamiltonians of N Cartesian coordinates using
    a multi-state quasi-diabatic (QD) Hamiltonian.
    
    
    Attributes
    ----------
    Vij : list
        Vij[i][j] refers to the diabat/coupling grid between states i and j.
    NH : int
        The size of the Hamiltonian.
    NV : int
        The size of the coordinate grid.
    NS : int
        The number of states.
    vvar : list 
        The active coordinates.
    grids : list
        Grids (for active) and fixed scalar values (for inactive) 
        of each coordinate.
    vshape : tuple
        The shape of the direct-product coordinate grid.
    masses : ndarray
        The mass of each coordinate.
    hbar : float
        The value of :math:`\\hbar`. 
    """
    
    def __init__(self, dvrs, pes, masses = None, hbar = None, pesorder = 'LR'):
        """
        
        Parameters
        ----------
        dvrs : list of GenericDVR objects and/or scalars
            A list of :class:`nitrogen.basis.GenericDVR` objects and/or
            scalar numbers. The length of the list must be equal to
            the number of coordinates, `nx`. Scalar elements indicate
            a fixed value for that coordinate.
        pes : DFun or function
            A potential energy/coupling surface function f(X) with respect
            to the `nx` Cartesian coordinates. This must have `NS`*(`NS`+1)/2
            output values for `NS` states.
        masses : array_like, optional
            A list of `nx` masses. If None, these will be assumed to be unity.
        hbar : float
            The value of :math:`\\hbar`. If None, the default value is that in 
            NITROGEN units. 
        pesorder : {'LR', 'LC'}, optional
            V matrix electronic state ordering. The `pes` function returns
            the *lower* triangle of the diabatic potential matrix. If 'LR'
            (default), the values are returned in row major order. If 'LC'
            the values are returned in column major order.
        """
        
        nx = len(dvrs)
        
        if masses is None:
            masses = [1.0 for i in range(nx)]
        
        if len(masses) != nx:
            raise ValueError("The number of masses must equal the length of dvrs")

        if hbar is None:
            hbar = nitrogen.constants.hbar 
    
        # Determine the active and fixed coordinates
        vvar = []
        grids = []
        vshape = []
        NV = 1
        D2list = []
        
        for i in range(len(dvrs)):
            
            if np.isscalar(dvrs[i]): # inactive
                grids.append(dvrs[i])
                vshape.append(1)
                D2list.append(None) 
            else: # active, assume DVR object
                vvar.append(i)
                grids.append(dvrs[i].grid)
                ni = dvrs[i].num # length of this dimension 
                vshape.append(ni) 
                NV *= ni 
                D2list.append(dvrs[i].D2)

        vshape = tuple(vshape)
        # To summarize:
        # -------------
        # vvar is a list of active coordinates
        # grids contains 1D active grids and fixed scalar values
        # vshape is the grid shape *including* singleton fixed coordinates
        # NV is the total grid size
        # D2list is the second-derivative operator list, including None's for fixed coord.
            
        if len(vvar) < 1:
            raise ValueError("There must be at least one active coordinate")
            
        # Calculate the coordinate grids
        Q = np.stack(np.meshgrid(*grids, indexing = 'ij'))
        
        # Calculate the PES/coupling grids
        try: # Attempt DFun interface
            V = pes.f(Q, deriv = 0)[0]  # shape is (NS*(NS+1)/2,) + vshape
        except:
            V = pes(Q) # Attempt raw function
            
        NS = round(np.floor(np.sqrt(2*V.shape[0])))
        
        # Make of list of individual grids
        Vij = [ [0 for j in range(NS)] for i in range(NS)]
        # Populate Vij list with references to individual grids
        if pesorder == 'LR':
            k = 0
            for i in range(NS):
                for j in range(i+1):
                    Vij[i][j] = V[k] # view of sub-array
                    Vij[j][i] = Vij[i][j] # Diabatic symmetry
                    k += 1
                    
        else: # pesorder = 'LC'
            k = 0
            for j in range(NS):
                for i in range(j+1):
                    Vij[i][j] = V[k] # view of sub-array
                    Vij[j][i] = Vij[i][j] # Diabatic symmetry
                    k += 1
        # Vij[i][j] now refers to the correct diabat/coupling surface
        
        NH = NS * NV  # The total size of the Hamiltonian
        
        # Determine the operator data-type
        dtype = np.result_type(Q,V)
         
        # Initialize the LinearOperator
        #super().__init__((NH,NH), matvec = self._cartn_mv, dtype = dtype)
        # Define the required LinearOperator attributes
        self.shape = (NH,NH)
        self.dtype = dtype
        
        # Define new attributes
        self.NH = NH
        self.NV = NV 
        self.NS = NS
        self.vvar = vvar 
        self.grids = grids 
        self.vshape = vshape 
        self._D2list = D2list 
        self.masses = np.array(masses)
        self.hbar = hbar 
        self.Vij = Vij
        
    
    def _matvec(self, x):
        
        evshape = (self.NS,) + self.vshape # Elec.-vib. tensor shape
        
        xgrid = x.reshape(evshape)  # Reshape vector to elec-vib grid
        y = np.zeros_like(xgrid)    # Result grid
        
        hbar = self.hbar 
        
        ############################################
        #
        for j in range(self.NS): # input block j
            xj = xgrid[j] # The coordinate grid of the j^th electronic block
            
            for i in range(self.NS): # output block i
                #######################################
                #
                # PES/coupling term (i <--> j)
                y[i] += self.Vij[i][j] * xj 
                #
                #
                # KEO (diagonal blocks only)
                if i == j:
                    y[i] += (-hbar**2 / 2.0) * dvrops.opO_coeff(xj, self._D2list, 1.0/self.masses)
                #
                #
                #######################################
        #        
        #
        ############################################
                    
        # Reshape result grid to a 1D vector
        return y.reshape((self.NH,)) 