from .coordsys import CoordSys, CoordTrans
import nitrogen.autodiff.forward as adf
import nitrogen.dfun as dfun
import numpy as np

__all__ = ['Valence3','CartesianN','LinearTrans']

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
                         maxderiv = None, isatomic = True,
                         zlevel = None)
        
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

    def __repr__(self):
        return f"Valence3({self.name!r})"
    
    def diagram(self):
        # using U+250X box and U+219X arrows
        diag = ""
        
        diag += "     │↓              ↑│        \n"
        diag += "     │Q [3]      [9] X│        \n"
        diag += "   ╔═╪════════════════╪═╗      \n"
        diag += "   ║ │ ┌────────────┐ │ ║      \n"
        diag += "   ║ ╰─┤ 3-atom val ├─╯ ║      \n"
        diag += "   ║   └────────────┘   ║      \n"
        diag += "   ╚════════════════════╝      \n"
        
        return diag
    
class CartesianN(CoordSys):
    """
    Cartesian coordinates in N-D space.
    
    :math:`X_i = Q_i`
    
    """
    
    def __init__(self, N):
        
        """
        Create a new CartesianN coordinate system object.
        
        Parameters
        ----------
        N : int
            The number of Cartesian coordinates.
        
        """
        
        name = f"{N:d}-D Cartesian"
        Qstr = [f"X{i:d}" for i in range(N)]
        Xstr = [f"X{i:d}" for i in range(N)]
        
        super().__init__(self._csCartN_q2x, nQ = N, 
                         nX = N, name = name, 
                         Qstr = Qstr, Xstr = Xstr,
                         maxderiv = None, isatomic = False,
                         zlevel = 1) # zlevel = 1 -- linear function
        
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
    
    def __repr__(self):
        return f'CartesianN({self.nQ!r}, {self.name!r})'
    
    def diagram(self):
        
        if len(self.name) > 15:
            name = self.name[:12] + "..."
        else:
            name = self.name
        
        # using U+250X box and U+219X arrows
        diag = ""
        
        sQ =f"[{self.nQ:d}]"
        sX =f"[{self.nX:d}]"
        
        diag += "     │↓              ↑│        \n"
        diag +=f"     │Q {sQ:<5s}  {sX:>5s} X│        \n"
        diag += "   ╔═╪════════════════╪═╗      \n"
        diag += "   ║ ╰────────────────╯ ║      \n"
        diag += "   ║   {:15s}  ║      \n".format(name)
        diag += "   ╚════════════════════╝      \n"
        
        return diag
            
class LinearTrans(CoordTrans):
    
    """
    Linear coordinate transformation.
    
    The output coordinates Qi are defined as
    
        :math:`Q_i = T_{ij} Q'_j`
        
    Parameters
    ----------
    
    T : ndarray
        The transformation matrix.
        
    """
    
    def __init__(self, T, Qpstr = None, name = None):
        """
        Create a LinearTrans object.

        Parameters
        ----------
        T : ndarray
            A square matrix.
        Qpstr: list of str, optional
            Labels for the new coordinates.

        """
        if np.ndim(T) != 2:
            raise ValueError("T must be 2-dimensional")
        m,n = T.shape
        if m != n:
            raise ValueError("T must be square")
            
        super().__init__(self._lintrans, nQp = m, nQ = m, 
                         name = name,
                         Qpstr = Qpstr, maxderiv = None,
                         zlevel = 1)
            
        self.T = T.copy()   # Transformation matrix, copy
        
    def _lintrans(self, Qp, deriv = 0, out = None, var = None):
        """
        Qp : ndarray
            Shape (self.nQp, ...)
        out : ndarray
            Shape (nd, self.nQ, ...)
        """
        
        base_shape =  Qp.shape[1:]
        N = self.nQp # = self.nX, number of inputs = number of outputs
        
        if var is None:
            var = [i for i in range(N)] # Calculate derivatives for all Qp
    
        nvar = len(var)
        nd = dfun.nderiv(deriv, nvar)
        
        if out is None:
            out = np.ndarray( (nd, N) + base_shape, dtype = Qp.dtype)
        
        out.fill(0) # Initialize derivative array to 0
        
        # 0th derivatives = values
        # Q_i = T_ij * Q'_j
        np.copyto(out[0],  np.tensordot(self.T, Qp, axes = (1,0)) )
        
        # 1st derivatives
        if deriv >= 1:
            for i in range(nvar):
                # derivatives with respect to k = var[i]
                # This is just the k^th column of T 
                for j in range(self.nQ):
                    out[i+1, j:(j+1)].fill(self.T[j,var[i]])
                
            
        # All higher derivatives are zero
        # zlevel reflects this, which other functions can
        # check to maximize efficiency.
        
        return out
    
    def __repr__(self):
        return f'LinearTrans({self.T!r}, Qpstr = {self.Qpstr!r})'
    
    def diagram(self):
        """ CoordTrans diagram string """
        
        sQ =f"[{self.nQ:d}]"
        sQp =f"[{self.nQp:d}]"
        
        if self.name is None:
            label = " T @ Q' "
        else:
            label = self.name 
            if len(label) > 8:
                label = label[:8]
            
        
        diag = ""
        
        diag += "     │↓       \n"
        diag +=f"     │Q'{sQp:<5s} \n"
        diag += "   ╔═╧══════╗ \n"
        diag += "   ║        ║ \n"
        diag +=f"   ║{label:^8s}║ \n"
        diag += "   ║        ║ \n"
        diag += "   ╚═╤══════╝ \n"
        diag +=f"     │Q {sQ:<5s} \n"
        
        return diag
