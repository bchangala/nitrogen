"""
nitrogen.coordsys
-----------------

This module implements the CoordSys base class,
which is extended for all NITROGEN coordinate
systems.

"""

import numpy as np
import nitrogen.dfun as dfun
import nitrogen.autodiff.forward as adf



class CoordSys(dfun.DFun):
    """
    A generic coordinate system base class, which
    extends the dfun.DFun generical differentiable
    function class.
    
    Attributes
    ----------
    nQ : int
        The number of input (curvlinear) coordinates.
    nX : int
        The number of output (Cartesian-like) coordinates.
    name : str
        The coordinate system label.
    Qstr : list of str
        The labels of the input coordinates.
    Xstr : list of st
        The labels of the output coordinates.
    maxderiv : int
        The maximum supported derivative order. :attr:`maxderiv` = -1 if 
        arbitrarily high-order derivatives are supported.
    isatomic : bool
        If :attr:`isatomic` == True, the coordinate system is *atomic*.
    natoms : int
        For atomic coordinate systems, the number of atoms. :attr:`nX`
        is equal to ``3*natoms``
    
    """
    
    def __init__(self, Q2Xfun, nQ = 1, nX = 1, name = '(unnamed coordinate system)',
                 Qstr = None, Xstr = None, maxderiv = -1, isatomic = False):
        """
        Create a new CoordSys object.

        Parameters
        ----------
        Q2Xfun : function
            An instance method for calculating X as a function of Q, the 
            coordinates. This function should have a signature
            ``QX2fun(self, Q, deriv = 0, out = None, var = None)``.
            See :meth:`CoordSys.Q2X` for details.
        nQ : int, optional
            Number of input (curvilinear) coordinates. The default is 1.
        nX : int, optional
            Number of output (Cartesian-like) coordinates. The default is 1.
        name : str, optional
            Coordinate system label. The default is '(unnamed coordinate system)'.
        Qstr : list of str, optional
            Labels for each of the `nQ` input coordinates. If None, these will be 
            automatically numbered.
        Xstr : list of str, optional
            Labels for each of the `nX` output coordinates. If None, these will be 
            automatically numbered.
        maxderiv : int, optional
            The maximum supported derviative order. The default is -1, indicating
            arbitrarily high derivatives are supported.
        isatomic : bool, optional
            If `isatomic` == True, then `nX` must be a multiple of 3, and the
            Cartesian-like output coordinates X should be ordered as
            :math:`(x_0, y_0, z_0, x_1, y_1, z_1,...)`. The default is False.

        """
        
        # Call DFun constructor
        super().__init__(Q2Xfun, nf = nX, nx = nQ, maxderiv = maxderiv)
        self.nQ = nQ
        self.nX = nX 
        self.name = name 
        
        if self.nQ < 1 :
            raise ValueError('nQ must be >= 1')
        if self.nX < 1 :
            raise ValueError('nX must be >= 1')
        
        # Q labels
        if Qstr is None:
            self.Qstr = ['Q%d' % i for i in range(self.nQ)]
        elif len(Qstr) != self.nQ :
            raise ValueError('Length of Qstr list must equal nQ')
        else:
            self.Qstr = Qstr
            
        self.isatomic = isatomic
        if self.isatomic:
            if self.nX % 3 == 0:
                self.natoms = self.nX // 3 # integer division
            else:
                raise ValueError('nX must be a multiple of 3 if isatomic == True')
        else:
            self.natoms = 0 # A non-atomic coord. sys. does not use `natoms`
        
        # X labels
        if Xstr is None:
            if self.isatomic:
                temp = [['x{0}'.format(i),'y{0}'.format(i),'z{0}'.format(i)] for i in range(self.natoms) ]
                temp = [val for sub in temp for val in sub]
                self.Xstr = temp # ['x0', 'y0', 'z0', 'x1', 'y1', 'z1', ...]
            else:
                self.Xstr = ['X%d' % i for i in range(self.nX)] # ['X0', 'X1', 'X2', ...]
        elif len(Xstr) != self.nX:
            raise ValueError('Length of Xstr list must equal nX') 
        else:
            self.Xstr = Xstr
            
            
    def Q2X(self, Q, deriv = 0, out = None, var = None):
        """
        Evaluate the coordinate function X(Q).

        Parameters
        ----------
        Q : ndarray
            An array of ``m`` input coordinate vectors.
            Q has shape (:attr:`nQ`, ``m``).
        deriv : int, optional
            All derivatives up through order `deriv` are requested. The default is 0.
        out : ndarray, optional
            Output location. If None, a new ndarray will
            be created. The default is None.
            See :attr:`nitrogen.dfun.DFun.f.out` for details.
        var : list of int, optional
            Variables with respect to which derivatives are taken. If None,
            all `nQ` variables will be used in the input order. The default is None.
            See :attr:`nitrogen.dfun.DFun.f.var` for details.

        Returns
        -------
        out : ndarray
            The result, X(Q).

        """
        # Check the requested derivative order
        if deriv < 0:
            raise ValueError('deriv must be non-negative')
        if deriv > self.maxderiv and self.maxderiv != -1:
            raise ValueError('deriv is larger than maxderiv')
        
        # Check the shape of input Q
        m,n = Q.shape
        if n != self.nQ:
            raise ValueError('Q must have shape (nQ, m)')
        
        # Check var
        if var is None:
            nvar = self.nQ  # Use all variables. None will be passed to _feval.
        else:
            if np.unique(var).size != len(var):
                raise ValueError('var cannot contain duplicate elements')
            if min(var) < 0 or max(var) >= self.nQ:
                raise ValueError('elements of var must be >= 0 and < nQ')
            nvar = len(var)
            
        # Create output array if no output buffer passed
        if out is None:
            nd = adf.nck(deriv + nvar, min(deriv,nvar))
            out = np.ndarray((self.nX, nd, m), dtype = Q.dtype)
            
        self._feval(self, Q, deriv, out, var) # Evaluate Q2X function
         
        return out
    
class CS_Valence3(CoordSys):
    """
    A triatomic valence coordinate system.
    
    The coordinates are :math:`r_1`, :math:`r_2`, and :math:`\\theta`.
    
    
    The first atom is at :math:`(0, 0, -r_1)`.
    The second atom is at the origin.
    The third atom is at :math:`(0, r_2 \sin\\theta, -r_2 \cos\\theta )`.
    
    """
    
    def __init__(self, name = 'Triatomic valence'):
        
        """
        Create a new CS_Valence3 object.
        
        Parameters
        ----------
        name : str, optional
            The coordinate system name. The default is 'Triatomic valence'.
        
        """
        
        super().__init__(self._csv3_q2x, nQ = 3, 
                         nX = 1, name = name, 
                         Qstr = ['r1', 'r2', 'theta'],
                         maxderiv = -1, isatomic = True)
        
    def _csv3_q2x(self, Q, deriv = 0, out = None, var = None):
        """
        Triatomic valence coordinate system Q2X instance method.
        See :meth:`CoordSys.Q2X` for details.
        
        Parameters
        ----------
        Q : ndarray
            Shape (self.nQ, m)
        deriv, out, var :
            See :meth:`CoordSys.Q2X` for details.
        
        Returns
        -------
        out : ndarray
            Shape (self.nX, nd, m)

        """
        
        natoms = 3 
        _,m = Q.shape 
        
        if var is None:
            var = [0, 1, 2] # Calculate derivatives for all Q
        
        nvar = len(var)
        
        nd = adf.nck(deriv + nvar, min(deriv, nvar)) # The number of derivatives
        
        # Create adf symbols/constants for each coordinate
        q = [] 
        for i in range(self.nQ):
            if i in var: # Derivatives requested for this variable
                q.append(adf.sym(Q[i], var.index(i), deriv, nvar))
            else: # Derivatives not requested, treat as constant
                q.append(adf.const(Q[i], deriv, nvar))
        # q = r1, r2, theta
        
        if out is None:
            out = np.ndarray( (3*natoms, nd, m), dtype = Q.dtype)
        out.fill(0) # Initialize out to 0
        
        # Calculate Cartesian coordinates
        np.copyto(out[2], (-q[0]).d ) # -r1
        np.copyto(out[7], (q[1] * adf.sin(q[2])).d ) #  r2 * sin(theta)
        np.copyto(out[8], (-q[1] * adf.cos(q[2])).d ) # -r2 * cos(theta)
        
        return out
        
        