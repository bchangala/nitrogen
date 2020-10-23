from .coordsys import CoordSys 
import nitrogen.autodiff.forward as adf
import nitrogen.dfun as dfun
import numpy as np

__all__ = ['Valence3','CartesianN']

class Valence3(CoordSys):
    """
    A triatomic valence coordinate system.
    
    The coordinates are :math:`r_1`, :math:`r_2`, and :math:`\\theta`.
    The first atom is at :math:`(0, 0, -r_1)`.
    The second atom is at the origin.
    The third atom is at :math:`(0, r_2 \sin\\theta, -r_2 \cos\\theta )`.
    
    """
    
    def __init__(self, name = 'Triatomic valence'):
        
        """
        Create a new Valence3 object.
        
        Parameters
        ----------
        name : str, optional
            The coordinate system name. The default is 'Triatomic valence'.
        
        """
        
        super().__init__(self._csv3_q2x, nQ = 3, 
                         nX = 9, name = name, 
                         Qstr = ['r1', 'r2', 'theta'],
                         maxderiv = -1, isatomic = True)
        
    def _csv3_q2x(self, Q, deriv = 0, out = None, var = None):
        """
        Triatomic valence coordinate system Q2X instance method.
        See :meth:`CoordSys.Q2X` for details.
        
        Parameters
        ----------
        Q : ndarray
            Shape (self.nQ, ...)
        deriv, out, var :
            See :meth:`CoordSys.Q2X` for details.
        
        Returns
        -------
        out : ndarray
            Shape (nd, self.nX, ...)

        """
        
        natoms = 3 
        base_shape =  Q.shape[1:]
        
        if var is None:
            var = [0, 1, 2] # Calculate derivatives for all Q
        
        nvar = len(var)
        
        # nd = adf.nck(deriv + nvar, min(deriv, nvar)) # The number of derivatives
        nd = dfun.nderiv(deriv, nvar)
        
        # Create adf symbols/constants for each coordinate
        q = [] 
        for i in range(self.nQ):
            if i in var: # Derivatives requested for this variable
                q.append(adf.sym(Q[i], var.index(i), deriv, nvar))
            else: # Derivatives not requested, treat as constant
                q.append(adf.const(Q[i], deriv, nvar))
        # q = r1, r2, theta
        
        if out is None:
            out = np.ndarray( (nd, 3*natoms) + base_shape, dtype = Q.dtype)
        out.fill(0) # Initialize out to 0
        
        # Calculate Cartesian coordinates
        np.copyto(out[:,2], (-q[0]).d ) # -r1
        np.copyto(out[:,7], (q[1] * adf.sin(q[2])).d ) #  r2 * sin(theta)
        np.copyto(out[:,8], (-q[1] * adf.cos(q[2])).d ) # -r2 * cos(theta)
        
        return out

class CartesianN(CoordSys):
    """
    Cartesian coordinates in N-D space.
    
    """
    
    def __init__(self, N, name = None):
        
        """
        Create a new CartesianN coordinate system object.
        
        Parameters
        ----------
        N : int
            The number of Cartesian coordinates.
        name : str, optional
            The coordinate system name. If None, this will be
            automatically created. The default is None.
        
        """
        
        if name is None:
            name = f"{N:d}-D Cartesian"
        Qstr = [f"X{i:d}" for i in range(N)]
        Xstr = [f"X{i:d}" for i in range(N)]
        
        super().__init__(self._csCartN_q2x, nQ = N, 
                         nX = N, name = name, 
                         Qstr = Qstr, Xstr = Xstr,
                         maxderiv = -1, isatomic = False)
        
    def _csCartN_q2x(self, Q, deriv = 0, out = None, var = None):
        """
        Q : ndarray
            Shape (self.nQ, ...)
        """
        
        base_shape =  Q.shape[1:]
        N = self.nQ # = self.nX
        
        if var is None:
            var = [i for i in range(N)] # Calculate derivatives for all Q
    
        nvar = len(var)
        nd = dfun.nderiv(deriv, nvar)
        
        if out is None:
            out = np.ndarray( (nd, N) + base_shape, dtype = Q.dtype)
        
        out.fill(0) # Initialize derivative array to 0
        
        # 0th derivatives
        # All X values are equal to input Q values
        np.copyto(out[0], Q)
        
        # 1st derivatives
        # All requested `var` have a derivative of 1 w.r.t its output
        for i in range(len(var)):
            out[1+i, var[i]:(var[i]+1)].fill(1.0)
            
        # All higher derivatives are zero
        
        return out
            
            
            
            
            
            