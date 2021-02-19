"""
jacobi.py

Jacobi coordinate systems

"""
from .coordsys import CoordSys 
import nitrogen.autodiff.forward as adf
import nitrogen.dfun as dfun
import numpy as np

__all__ = ['JacobiChain3N'] # imported by import *

class JacobiChain3N(CoordSys):
    
    """
    Simple chained Cartesian Jacobi coordinates 
    for N particles in a space-fixed 3D frame.
    
    The Jacobi binary tree diagram is
    
    ::
    
           +
          / \
         0   +
            / \
           1   +
              / \
             2   + 
                ...
        
      
    The first three coordinates are the Cartesian
    position of the total center-of-mass. 
    The second three coordinates are the Jacobi
    vector from atom 0 to its sibling node, and so on.  
    
    
    Attributes
    ----------
    
    jacobi_masses : ndarray
        The masses defining the Jacobi coordinates
    mu : ndarray
        The effective mass of each Jacobi vector
    J : ndarray
        The linear transformation matrix between
        the Jacobi coordinates and space-frame
        Cartesian coordinates (X = J*Q)
    
    
    """
    
    
    def __init__(self, jacobi_masses):
        """
        

        Parameters
        ----------
        jacobi_masses : array_like
            An array of N particles masses
            defining the Jacobi coordinates. (These are
            *not* the masses used for calculating metric
            tensors.)

        """
    
        # Create a copy of jacobi_masses for instance variable
        jacobi_masses = np.array(jacobi_masses).copy()
        natoms = jacobi_masses.size  # The number of atoms
        mu = np.empty((natoms,)) # Effective masses 
        rm = np.empty((natoms-1,2)) # Scaling coefficients for Jacobi r-vectors
        
        if natoms < 2:
            raise ValueError("At least 2 atoms are required")
        
        ##########################################
        # Calculate the effective masses
        # for each Jacobi vector, working
        # backwards up the tree
        mR = jacobi_masses[-1] # The mass of the last atom
        for i in range(natoms-1, 0, -1):
            mL = jacobi_masses[i-1]         # Left branch mass
            mu[i] = (mL * mR) / (mL + mR)   # Reduced mass 
            
            rm[i-1,0] = -mR / (mL + mR)       # Left branch
            rm[i-1,1] = +mL / (mL + mR)       # Right branch
            
            mR = mL + mR                    # Virtual mass of new node
        #
        # The top-most effective mass is the total mass
        mu[0] = np.sum(jacobi_masses) 
        ############################################
        
        ############################################
        # Calculate the linear Jacobi transformation matrix
        J = np.zeros((natoms, natoms)) # Linear transformation matrix 
        #
        #    [ x0 ]    [       ][ r0 ]  (r0 is COM)
        #    [ x1 ]  = [   J   ][ r1 ]
        #    [ x2 ]    [       ][ r2 ]
        #
        
        for i in range(natoms):
            J[i,0] += 1.0  # Center of mass motion
        
        for j in range(natoms-1):
            # Effect of the i**th relative vector
            J[j,j+1] += rm[j,0] 
            
            for i in range(j+1,natoms):
                J[i,j+1] += rm[j,1] 
        
        Jfull = np.kron(J, np.eye(3)) # Expand to 3-dimensions
        
        
        #######################################
        # Initialize CoordSys
        super().__init__(self._jac_chain_3n, 3*natoms, 3*natoms, name = 'Jac. Chain 3N',
                         isatomic = True, zlevel = 1)
        # Note that zlevel is set to 1.
        
        
        # new attributes
        self.jacobi_masses = jacobi_masses 
        self.mu = mu 
        self._rm = rm
        self.J = Jfull
        
    
    def _jac_chain_3n(self, Q, deriv = 0, out = None, var = None):
        """
        Q(X) function

        Parameters
        ----------
        Q : ndarray
            Input coordinate array (`self.nQ`, ...)
        deriv : int, optional
            Maximum derivative order. The default is 0.
        out : ndarray, optional
            Output location (nd, `self.nX`, ...) . The default is None.
        var : list of int, optional
            Requested ordered derivative variables. If None, all are used.

        Returns
        -------
        out : ndarray
            The X coordinates.

        """
        
        base_shape =  Q.shape[1:]
        N = self.nQ
        
        if var is None:
            var = [i for i in range(N)] # Calculate derivatives for all Qp
    
        nvar = len(var)
        nd = dfun.nderiv(deriv, nvar)
        
        if out is None:
            out = np.ndarray( (nd, N) + base_shape, dtype = Q.dtype)
        
        out.fill(0) # Initialize derivative array to 0
        
        # 0th derivatives = values
        # X_i = J_ij * Q_j 
        np.copyto(out[0],  np.tensordot(self.J, Q, axes = (1,0)) )
  
        # 1st derivatives
        if deriv >= 1:
            for i in range(nvar):
                # derivatives with respect to k = var[i]
                # This is just the k^th column of J
                for j in range(self.nQ):
                    out[i+1, j:(j+1)].fill(self.J[j,var[i]])
                
        # All higher derivatives are zero
        # zlevel reflects this, which other functions can
        # check to maximize efficiency.
        
        return out
    
    def __repr__(self):
        return f'JacobiChain3N({self.jacobi_masses!r})'
    
    def diagram(self): 
        # using U+250X box and U+219X arrows
        diag = ""
        sQ =f"[{self.nQ:d}]"
        sX =f"[{self.nX:d}]"
        
        diag += "     │↓              ↑│        \n"
        diag +=f"     │Q {sQ:<5s}  {sX:>5s} X│        \n"
        diag += "   ╔═╪════════════════╪═╗      \n"
        diag += "   ║ │ ┌────────────┐ │ ║      \n"
        diag += "   ║ │ │   Jacobi   │ │ ║      \n"
        diag += "   ║ │ │  Chain 3N  │ │ ║      \n"
        diag += "   ║ │ │            │ │ ║      \n"
        diag += "   ║ ╰─┤   ╱╲       ├─╯ ║      \n"
        diag += "   ║   │  0 ╱╲      │   ║      \n"
        diag += "   ║   │   1 ╱╲     │   ║      \n"
        diag += "   ║   │    2 ╱╲    │   ║      \n"
        diag += "   ║   │     3 ...  │   ║      \n"
        diag += "   ║   └────────────┘   ║      \n"
        diag += "   ╚════════════════════╝      \n"
        
        return diag
       