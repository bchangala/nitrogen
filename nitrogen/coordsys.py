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
    extends the dfun.DFun generic differentiable
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
            An array of input coordinates with shape (:attr:`nQ`, ...).
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
            The result, X(Q), in DFun format.

        """
        # Check the requested derivative order
        if deriv < 0:
            raise ValueError('deriv must be non-negative')
        if deriv > self.maxderiv and self.maxderiv != -1:
            raise ValueError('deriv is larger than maxderiv')
        
        # Check the shape of input Q
        n = Q.shape[0]
        if n != self.nQ:
            raise ValueError('Q must have shape (nQ, ...)')
        
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
            out = np.ndarray((nd, self.nX) + Q.shape[1:], dtype = Q.dtype)
            
        self._feval(Q, deriv, out, var) # Evaluate Q2X function
         
        return out
    
    def Q2t(self, Q, deriv = 0, out = None, vvar = None, rvar = None):
        """
        Calculate t-vectors and their derivatives
        (for *atomic* :class:`CoordSys` objects).

        Parameters
        ----------
        Q : ndarray
            An array of ``m`` input coordinate vectors.
            Q has shape (:attr:`nQ`, ...).
        deriv : int
            Derivative level. The default is 0.
        out : ndarray, optional
            Output location, an ndarray with shape ``(nd, nt, natoms, 3, ...)``
            and the same data-type as Q. If None, this will be created.
            The default is None.
        vvar : list of int, optional
            The coordinates for which vibrational t-vectors, and derivatives
            thereof, will be calculated. If None, then all coordinates will 
            be used in order. The default is None.
        rvar : str, optional
            The body-fixed axes for which rotational t-vectors will be calculated.
            This is specified by a string containing 'x', 'y', and 'z', e.g.
            'xyz' or 'zy'. '' will calculate no rotational t-vectors, and None
            is equivalent to 'xyz'. The default is None.

        Returns
        -------
        out : ndarray
            The t-vector array with shape ``(nd, nt, natoms, 3, ...)``. The second index
            runs over the requested vibrational and rotational coordinates in 
            the order given by `vvar` and `rvar`. The first dimension runs over
            the `nd` requested derivatives in normal DFun order for the number of
            independent variables indicated by `vvar`.
            

        """
        
        if not self.isatomic: 
            raise RuntimeError('Q2t is only callable for atomic CoordSys objects')
        
        if vvar is None:
            vvar = [i for i in range(self.nQ)] # Calculate all vibrational t-vectors
            
        if rvar is None:
            rvar = 'xyz'
        if not rvar in ['', 'x', 'y', 'z', 'xy', 'yx', 'xz','zx','yz','zy',
                        'xyz','yzx','zxy','zyx','yxz','xzy']:
            raise ValueError('Invalid rvar string')
        
        nv = len(vvar)  # The number of vibrational coordinates
        nr = len(rvar)  # The number of rotational coordinates
        nt = nv + nr    # The total number of t-vectors
        na = self.natoms # The number of atoms
        nd = dfun.nderiv(deriv, nv) # The number of derivatives
        base_shape = Q.shape[1:]
        
        if nt == 0:
            raise ValueError('At least vvar or rvar must be non-empty')
        
        # Calculate the Cartesian coordinates and derivatives
        # We need one higher order than deriv
        X = self.Q2X(Q, deriv = deriv + 1, var = vvar)
        
        if out is None:
            out = np.ndarray((nd, nt, na, 3) + base_shape, Q.dtype)
        
        
        # Extract vibrational t-vectors
        if nv > 0:
            idxtab = adf.idxtab(deriv+1, nv) # needed for order reduction
            
            for k in range(nv): # For each vibrational coordinate k
                for i in range(na): # For each atom i
                    # 
                    # X[:, 3*i:(3*i+3), ...] is the derivative array
                    # for the i^th atom's Cartesian position
                    #
                    # The k^th vibrational t-vector is the derivative of this with
                    # respect to the vibrational coordinate k
                    #
                    # Extract the k^th derivative of X from its super
                    # derivative array
                    adf.reduceOrder(X[:, 3*i:(3*i+3)], k, deriv+1, nv, idxtab, 
                                    out = out[:, k, i, :])
                    # (note that the base_shape is implicitly handled)
          
        # Calculate rotational t-vectors
        ea = np.eye(3) 
        xyz = {'x':0, 'y':1, 'z':2}
        
        for a in range(nr): # For each requested axis 
            alpha = xyz[rvar[a]] # the axis index (0, 1, or 2 for x, y, or z)
            for i in range(na): # For each atom i
                # The rotational t-vector is the derivative
                # of atom i with respect to an infinitesimal
                # rotation about axis alpha. This is equal to
                # the cross-product of the unit-vector along
                # alpha and the position vector of atom i
                # 
                np.copyto( out[:,-nr+a, i, :], 
                          np.cross(ea[:,alpha], X[0:nd, (3*i):(3*i+3)], 
                                   axisa = 0,axisb = 1, axisc = 1))
                # Note the use of `axis` in np.cross !!!
                # (the base_shape is implicitly handled)
        
        return out
    
    def Q2g(self, Q, masses = None, deriv = 0, out = None, vvar = None, rvar = None,
            mode = 'bodyframe'):
        """
        Calculate the curvilinear metric tensor g

        Parameters
        ----------
        Q : ndarray
            An array of input coordinates.
            Q has shape (:attr:`nQ`, ...).
        masses : array_like
            A list of masses of length ``natoms``.
        deriv : int
            The requested derivative order. Derivatives are calculated 
            for all vibrational coordinates indicated by `vvar`. The default is 0.
        out : ndarray, optional
            Output location, an ndarray with shape ``(nd, ng, ...)``
            and the same data-type as Q. If None, this will be created.
            The default is None.
        vvar : list of int, optional
            The coordinates included in the vibrational block of g.
            If None, then all coordinates will be used in order. The default is None.
        rvar : str, optional
            The body-fixed axes inlcuded in the rotational block of g.
            If None, then all axes will be used in order. The default is None.
        mode : {'bodyframe'}
            Calculation mode. 'bodyframe' calculates the standard g tensor
            for a rotating molecule (CoordSys must be *atomic*).

        Returns
        -------
        out : ndarray
            The g tensor derivative array with shape ``(nd, ng, ...)`` 
            stored in packed upper triangle column-major order.

        """
        
        if mode == 'bodyframe':
            if not self.isatomic:
                raise ValueError("'bodyframe' mode is valid for atomic CoordSys only")
            if masses is None:
                raise ValueError("masses must be specified for 'bodyframe' mode")
            if len(masses) != self.natoms:
                raise ValueError("length of masses must equal natoms")
            
            # Determine the number of vibrational coordinates
            if vvar is None:
                nv = self.nQ 
            else:
                nv = len(vvar)
            # Calculate the t-vectors
            t = self.Q2t(Q, deriv=deriv, vvar = vvar, rvar = rvar)
            
            nd = t.shape[0]
            nt = t.shape[1]
            base_shape = Q.shape[1:]
            
            ng = (nt*(nt+1))//2
            
            if out is None:
                out = np.ndarray((nd,ng)+base_shape, dtype = Q.dtype)
            # Calculate the g metric
            self.t2g(t, masses, deriv, nv, fixCOM=True, out = out)
            
        else:
            raise ValueError('Invalid mode string')
            
        return out
    
    @staticmethod
    def t2g(t, masses, deriv, nv, fixCOM = True, out = None):
        """
        Calculate g metric tensor given atomic t-vectors
        and masses.
        
        Parameters
        ----------
        t : ndarray 
            Atomic t-vector array, with shape ``(nd, nt, natoms, 3, ...)``, 
            as returned by :meth:`Q2t`.
        masses : array_like
            A list of masses of length ``natoms``.
        deriv : int
            The derivative order of the t-vector derivative arrays
        nv : int
            The number of variables w.r.t which the derivative arrays were 
            calculated.
        fixCOM : bool
            If fixCOM, then the t vectors are shifted to 
            the center-of-mass frame before calculating the
            g metric tensor. **This modifies `t`.** The default is True.
        out : ndarray
            The output location with shape ``(nd, ng, ...)``, where
            ``ng = (nt * (nt+1)) // 2``. If None, this
            will be created. The default is None. 

        Returns
        -------
        out : ndarray 
            The curvlinear metric tensor in derivative array format.
            The second-index is in packed upper triangle column-major order.

        """
        nd = t.shape[0]
        nt = t.shape[1]
        natoms = t.shape[2]
        base_shape = t.shape[4:]
        
        if nd != dfun.nderiv(deriv, nv):
            raise ValueError("Inconsistent deriv or nv value")

        ng = (nt*(nt+1)) // 2  # Number of elements for packed storage of g
        
        if out is None:
            out = np.ndarray((nd,ng) + base_shape, dtype = t.dtype)
        out.fill(0) # Initialize result to zero
        
        
        if fixCOM: # Shift t-vectors to Center-of-Mass frame
            
            # First, calculate t-vector of COM in original frame
            tCOM = np.zeros((nd,nt,3)+base_shape)
            for a in range(natoms):
                tCOM += masses[a] * t[:,:,a,:]
            tCOM = tCOM / sum(masses)
            # Now, subtract COM from t-vectors
            for a in range(natoms):
                    t[:,:,a,:] = t[:,:,a,:] - tCOM
        
        #
        # Calculate g tensor
        #
        # This contains (dot) products of t-vectors, so 
        # we will use the generalized Leibniz routine to
        # compute the products of the derivative arrays for each
        # t-vector and then sum along the x/y/z dimension
        #
        idxtab = adf.idxtab(deriv, nv)
        ncktab = adf.ncktab(deriv+nv, min(nv,deriv))
        tt_val = np.ndarray((nd,3)+base_shape)
        
        idx = 0
        for j in range(nt):
            for i in range(j+1):
                for a in range(natoms):
                    tj = t[:,j,a,:] # 
                    ti = t[:,i,a,:] # shape = (nd, 3) + base_shape
                    
                    # Compute temp_val <-- tj * ti
                    adf.mvleibniz(tj, ti, deriv, nv, ncktab, idxtab, out=tt_val)
                    # Sum over the x/y/z/ axis dimension
                    out[:, idx] += masses[a] * np.sum(tt_val, axis = 1)
                
                idx += 1 
        
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

