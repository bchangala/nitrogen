import numpy as np
import nitrogen.dfun as dfun
import nitrogen.autodiff.forward as adf
import nitrogen.constants

__all__ = ["CoordSys", "CoordTrans", "CompositeCoordTrans",
           "QTransCoordSys"]

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
        The maximum supported derivative order. :attr:`maxderiv` = None if 
        arbitrarily high-order derivatives are supported.
    isatomic : bool
        If :attr:`isatomic` == True, the coordinate system is *atomic*.
    natoms : int
        For atomic coordinate systems, the number of atoms. :attr:`nX`
        is equal to ``3*natoms``
    
    """
    
    def __init__(self, Q2Xfun, nQ = 1, nX = 1, name = '(unnamed coordinate system)',
                 Qstr = None, Xstr = None, maxderiv = None, isatomic = False,
                 zlevel = None):
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
            The maximum supported derviative order. The default is None, indicating
            arbitrarily high derivatives are supported.
        isatomic : bool, optional
            If `isatomic` == True, then `nX` must be a multiple of 3, and the
            Cartesian-like output coordinates X should be ordered as
            :math:`(x_0, y_0, z_0, x_1, y_1, z_1,...)`. The default is False.
        zlevel : int, optional
            The zero-level of the differentiable Q2Xfun function. The default
            is None.

        """
        
        # Call DFun initializer
        super().__init__(Q2Xfun, nf = nX, nx = nQ, maxderiv = maxderiv, zlevel = zlevel)
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
            raise ValueError('Length of Qstr list ({:d}) must equal nQ ({:d})'.format(len(Qstr),self.nQ))
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
        # # Check the requested derivative order
        # if deriv < 0:
        #     raise ValueError('deriv must be non-negative')
        # if self.maxderiv is not None and deriv > self.maxderiv:
        #     raise ValueError('deriv is larger than maxderiv')
        
        # # Check the shape of input Q
        # n = Q.shape[0]
        # if n != self.nQ:
        #     raise ValueError('Q must have shape (nQ, ...)')
        
        # # Check var
        # if var is None:
        #     nvar = self.nQ  # Use all variables. None will be passed to _feval.
        # else:
        #     if np.unique(var).size != len(var):
        #         raise ValueError('var cannot contain duplicate elements')
        #     if min(var) < 0 or max(var) >= self.nQ:
        #         raise ValueError('elements of var must be >= 0 and < nQ')
        #     nvar = len(var)
            
        # # Create output array if no output buffer passed
        # if out is None:
        #     nd = dfun.nderiv(deriv, nvar)
        #     out = np.ndarray((nd, self.nX) + Q.shape[1:], dtype = Q.dtype)
            
        # self._feval(Q, deriv, out, var) # Evaluate Q2X function
        
        return self.f(Q, deriv, out, var)
    
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
            A list of masses of length ``natoms`` (if `mode` == 'bodyframe')
            or length ``nX`` (if `mode` == 'simple'). A value of None
            is interpreted as unit masses (for applicable modes).
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
            The body-fixed axes included in the rotational block of g.
            If None, then all axes will be used in order (rvar = 'xyz'). The default is None.
            This only applied to `mode` = 'bodyframe'.
        mode : {'bodyframe','simple'}
            Calculation mode. 'bodyframe' (default) calculates the standard g tensor
            for a rotating molecule (CoordSys must be *atomic*). 'simple' 
            assumes a rectangular (Cartesian) space-fixed embedding.

        Returns
        -------
        out : ndarray
            The g tensor derivative array with shape ``(nd, ng, ...)`` 
            stored in packed upper triangle column-major order.

        """
        
        if mode == 'bodyframe':
            #
            # Body-frame embedding for a system of particles.
            #
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
        
        elif mode == 'simple':
            #
            # A simple set of nX rectangular coordinates
            #
            if masses is None:
                # Default: unit masses
                masses = [1.0 for i in range(self.nX)]
            elif len(masses) != self.nX:
                raise ValueError("length of masses must equal nX")
            
            # Calculate the coordinate system Jacobian 
            s = self.jacderiv(Q, deriv=deriv, var = vvar) 
            
            nd = s.shape[0] 
            ns = s.shape[1] 
            base_shape = Q.shape[1:] 
            
            ng = (ns*(ns+1)) // 2 
            
            if out is None:
                out = np.ndarray((nd,ng) + base_shape, dtype = s.dtype)
            # Calculate g metric from s (the Jacobian)
            self.s2g(s, masses, deriv, out = out)
            
            
        else:
            raise ValueError('Invalid mode string')
            
        return out
    
    @staticmethod 
    def s2g(s, masses, deriv, out = None):
        """
        Calculate g metric tensor for a simple embedding.

        Parameters
        ----------
        s : ndarray
            An (nd, nv, nX, ...) array containing the coordinate
            system Jacobian derivative array.
        masses : array_like
            A list of masses of length `nv`
        deriv : int 
            The derivative order of the `s` array.
        out : ndarray, optional
            an (nd, ng, ...) output array, where
            ng = (nv * (nv + 1))

        Returns
        -------
        out : ndarray
            Result.

        """
        
        nd = s.shape[0]
        nv = s.shape[1] # The number of curvilinear coordinates
        nX = s.shape[2] # The number of rectangular coordinates
        base_shape = s.shape[3:] 
        
        if nd != dfun.nderiv(deriv, nv):
            raise ValueError("Inconsistent deriv or nv value")
        
        ng = (nv * (nv+1) ) // 2 # Number of element of packed storage of g
        
        if out is None: 
            out = np.ndarray( (nd, ng) + base_shape, dtype = s.dtype)
        out.fill(0.0) # Initialize to zero
        
        # Calculate g-tensor
        # using Leibniz product routine
        
        idxtab = adf.idxtab(deriv, nv)
        ncktab = adf.ncktab(deriv+nv, min(nv,deriv))
        sisj = np.ndarray( (nd,) + base_shape, dtype = s.dtype)
        
        idx = 0
        # Loop over curvilinear coordinates
        for j in range(nv):
            for i in range(j+1):
                # Sum over rectangular coordinates
                for a in range(nX):
                    si = s[:,i,a]
                    sj = s[:,j,a] # shape = (nd, ) + base_shape
                    # Compute sisj <-- si * sj
                    adf.mvleibniz(si,sj, deriv, nv, ncktab, idxtab, out = sisj)
                    # g_ij <-- m * si * sj
                    out[:, idx] += masses[a] * sisj 
                
                idx += 1 # increment packed storage index
        
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
        out : ndarray, optional
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
            tCOM = np.zeros((nd,nt,3)+base_shape, dtype = t.dtype)
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
        tt_val = np.ndarray((nd,3)+base_shape, dtype = t.dtype)
        
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
    
    def diagram(self):
        # using U+250X box and U+219X arrows
        diag = ""
        
        sQ =f"[{self.nQ:d}]"
        sX =f"[{self.nX:d}]"
        # generic coordinate system
        diag += "     │↓              ↑│        \n"
        diag +=f"     │Q {sQ:<5s}  {sX:>5s} X│        \n"
        diag += "   ╔═╪════════════════╪═╗      \n"
        diag += "   ║ │ ┌────────────┐ │ ║      \n"
        diag += "   ║ ╰─┤  CoordSys  ├─╯ ║      \n"
        diag += "   ║   └────────────┘   ║      \n"
        diag += "   ╚════════════════════╝      \n"
        
        return diag
    
    
    def __matmul__(self, other):
        """ self @ other
        
            This chains self(other(x))
        """
        if isinstance(other, CoordTrans):
            return QTransCoordSys(other, self)
        else:
            return super.__matmul__(other) 
        
    def Q2GUV(self, q, masses, deriv, var = None, hbar = None):
        """
        
        Compute the kinetic energy operator coefficients in 
        the pseudo-potential representation.
        
        Parameters
        ----------
        q : (nq,...) ndarray
            The evaluation grid.
        masses : array_like
            The masses.
        deriv : integer
            The derivative order of `G`. The derivative order of `U` and `V`
            will be one less.
        var : array_like, optional
            The active coordinates. If None (default), all are used.
        hbar : float, optional
            The value of :math:`\\hbar`. If None (default), standard
            NITROGEN units will be used. 

        Returns
        -------
        Gvib : (nd1, nvar, nvar, ...) 
            The vibrational block of :math:`G` multplied by :math:`\\hbar^2 / 2`.
        Grv : (nd1, 3, nvar, ...)
            The rotational-vibrational block of :math:`G` multplied by :math:`\\hbar^2 / 2`.
        Grot : (nd1, 3, 3, ...)
            The rotational block of :math:`G` multplied by :math:`\\hbar^2 / 2`.
        U : (nd2, nvar, ...)
            The :math:`U_i` functions multplied by :math:`\\hbar^2 / 2`.
        VT : (nd2, ...)
            The :math:`V_T` pseudo-potential.

        """
        
        dg = self.Q2g(q, masses = masses, deriv = deriv, vvar = var,
                    mode = 'bodyframe')
        
        nd, nvar = dfun.ndnvar(deriv, var, self.nx)
        
        # dg ... (nd, gpacked, ...)
        
        base_shape = q.shape[1:] # The base_shape 
        ng = nvar + 3  # The size of g, n x n
        
        #
        # Unpack to full matrices. Keep matrix dimensions
        # after derivative index 
        #
        dg_full = np.empty((nd,) + (ng,ng) + base_shape)
        # dg is in packed storage. Lower triangle, row major (= upper triangle, column major)
        idx = 0
        for i in range(ng):
            for j in range(i+1):
                np.copyto(dg_full[:,i,j], dg[:,idx])
                if i != j:
                    np.copyto(dg_full[:,j,i], dg[:,idx])
                idx += 1 
        # 
        # dg_full is complete.
        #
        # Compute the inverse G.
        # 
        
        # Use the more efficient `product_table` routines
        # from adf.
        #
        # The index table and product table for g and G 
        idxtab = adf.idxtab(deriv, nvar)
        dpt = adf.calc_product_table(deriv, nvar) 
        
        # Calculate the tables for deriv - 1 
        if deriv > 0:
            # U and VT will be calculated 
            idxtab_UV = adf.idxtab(deriv-1, nvar)
            dpt_UV = adf.calc_product_table(deriv-1, nvar) 
            nd_UV = idxtab_UV.shape[0]
        
        # 
        # Compute the rovibrational inverse metric 
        #
        dg_swap = np.moveaxis(dg_full, [1,2], [-2,-1])
        dG_full = np.empty_like(dg_swap)
        temp = np.empty_like(dg_swap[0])
        
        adf.linalg._inv_ad_table(dg_swap, dG_full, temp, dpt)
        
        dG_full = np.moveaxis(dG_full, [-2,-1], [1,2])
        
        #
        # dG_full contains the inverse metric 
        #
        # (nd, ng, ng, ...)
        #
        if hbar is None:
                hbar = nitrogen.constants.hbar 
        
        def trAB(dA,dB,out,dpt):
            # Tr[A @ B]
            # trace of matrix product of two matrices
            # the matrix axes are axis 1,2
            #
            m,n = dA.shape[1:3] 
            # A : m x n matrix 
            # B : n x m matrix 
            
            out.fill(0)
            for i in range(m):
                for j in range(n):
                    adf._mul_ad_table(out, dA[:,i,j], dB[:,j,i], dpt, reduce = True)
            return         
        
        if deriv > 0:
            # Extract the single derivatives of g to compute 
            # gtilde 
            gi = [adf.reduceOrder(dg_full, i, deriv, nvar, idxtab) for i in range(nvar)]
            # 
            # Compute gtilde_i = (d_i |g|) / |g| = Tr[G @ d_i g]
            #
            
            
            gtilde = [] 
            for i in range(nvar):
                out = np.empty_like(dg_full, shape = (nd_UV,) + base_shape)
                trAB(dG_full, gi[i], out, dpt_UV)
                gtilde.append(out)
            
            # Calculate the final derivative values
            
            # Gamma = -0.5 * dg/g
            Gammatilde = [-0.5 * e for e in gtilde]
            
            U = []
            for i in range(nvar):
                #
                # Calculate hbar**2/2 * Ui 
                Ui = 0.0 * Gammatilde[0] 
                
                for j in range(nvar):
                    # Ui +=  Gammatilde[j] * Gvib[i][j]
                    adf._mul_ad_table(Ui, Gammatilde[j], dG_full[:,i,j], dpt_UV, reduce = True)
                    
                Ui *= (hbar**2 / 4.0)
                U.append(Ui)
                
            U = np.stack(U, axis = 1) # Stack along axis = 1
            
            # 3) Scalar term (VT) 
            #
            VT = 0.0 * Gammatilde[0] 
            temp = 0.0 * VT 
            
            for i in range(nvar):
                for j in range(i+1):
                    
                    factor = hbar**2 / 8.0 
                    if i != j:
                        factor *= 2.0 
                    adf._mul_ad_table(temp, Gammatilde[i], Gammatilde[j], dpt_UV)
                    temp *= factor 
                    adf._mul_ad_table(VT, temp, dG_full[:,i,j], dpt_UV, reduce = True)
                    #  VT += factor * Gammatilde[i] * Gammatilde[j] * Gvib[i][j]
        else:
            U = None 
            VT = None 
            
        dG_full *= (hbar**2 / 2.0 )
        
        Gvib = dG_full[:, :nvar,:nvar]  # (vib,vib)
        Grv = dG_full[:, nvar:,:nvar]   # (rot,vib)
        Grot = dG_full[:, nvar:,nvar:]  # (rot,rot)

        return Gvib, Grv, Grot, U, VT  
    
class CoordTrans(dfun.DFun):
    """
    A base class for coordinate transformations.
    
    """
    
    def __init__(self, dfunction, nQp = 1, nQ = 1, 
                 name = '(unnamed coordinate transformation)',
                 Qpstr = None, Qstr = None, maxderiv = None, zlevel = None):
        """

        Parameters
        ----------
        dfunction : DFun or function
            A differentiable function defining the coordinate
            transformation Q(Q')
        nQp : int, optional
            The number of new (input) coordinates. Ignored if
            dfunction is a DFun. The default is 1.
        nQ : int, optional
            The number of old (output) coordinates. Ignored if
            dfunction is a DFun. The default is 1.
        name : str, optional
            Transformation name.
        Qpstr, Qstr : list of str, optional
            Coordinate labels
        maxderiv : int, optional
            The maximum supported derivative order. Ignored if
            dfunction is a DFun. The default is None (no maximum).
        zlevel : int, optional
            The zero-level of the Q(Q') DFun. Ignored if
            dfunction is a DFun. The default is None.

        """
        
        if isinstance(dfunction, dfun.DFun):
            super().__init__(dfunction._feval, nf=dfunction.nf,
                             nx=dfunction.nx, maxderiv = dfunction.maxderiv,
                             zlevel = dfunction.zlevel)
                
        else: # we expect a function
            # dfunction should have a signature
            # of (self, Qp, deriv = 0, out = None, var = None)
            super().__init__(dfunction, nf = nQ, nx = nQp,
                             maxderiv = maxderiv, zlevel = zlevel)
        
        self.nQp = self.nx # Inputs 
        self.nQ  = self.nf # Outputs 
        self.name = name        
        if Qpstr is None:
            self.Qpstr = [f"Q{i:d}'" for i in range(self.nQp)]
        else:
            self.Qpstr = Qpstr 
            
        if Qstr is None: 
            self.Qstr = [f"Q{i:d}" for i in range(self.nQ)]
        else: 
            self.Qstr = Qstr
    
    def Qp2Q(self, Qp, deriv = 0, out = None, var = None):
        """
        Evaluate the transformation function Q(Q')

        Parameters
        ----------
        Qp : ndarray
            An array of input coordinates with shape (:attr:`nQp`, ...).
        deriv : int, optional
            All derivatives up through order `deriv` are requested. The default is 0.
        out : ndarray, optional
            Output location. If None, a new ndarray will
            be created. The default is None.
        var : list of int, optional
            Variables with respect to which derivatives are taken. If None,
            all `nQp` variables will be used in the input order. The default is None.

        Returns
        -------
        out : ndarray
            The (nd, nQ, ...) derivate array for Q(Q'), in DFun format.

        """        
        return self.f(Qp, deriv, out, var)   # Evaluate the DFun.f function
    
    def diagram(self):
        """ CoordTrans diagram string """
        
        sQ =f"[{self.nQ:d}]"
        sQp =f"[{self.nQp:d}]"
        
        diag = ""
        # generic coordinate transformation
        diag += "     │↓       \n"
        diag +=f"     │Q'{sQp:<5s} \n"
        diag += "   ╔═╧═════╗  \n"
        diag += "   ║       ║  \n"
        diag += "   ║ Coord ║  \n"
        diag += "   ║ Trans ║  \n"
        diag += "   ║       ║  \n"
        diag += "   ╚═╤═════╝  \n"
        diag +=f"     │Q {sQ:<5s} \n"
        
        return diag
    
    def __pow__(self, other):
        """ self ** other
        
            This operation chains self --> other
        
        """
        if isinstance(other, CoordTrans):
            # Return a composite CoordTrans
            return CompositeCoordTrans(other, self)
        
        elif isinstance(other, CoordSys):
            # Return a transformed CoordSys
            return QTransCoordSys(self, other)
        
        else: 
            return super().__pow__(other)
    
    def __matmul__(self, other):
        """ self @ other 
            
            This operation chains self(other(x))
            
        """
        if isinstance(other, CoordTrans):
            return CompositeCoordTrans(self, other)
        else:
            return super().__matmul__(other)
        
class CompositeCoordTrans(CoordTrans):
    """ Composite coordinate transformation
    """
    
    def __init__(self, A, B):
        """ Q = A(B(Q')) """
        
        # First make a composite DFun
        df = dfun.CompositeDFun(A,B)
        # (This is to get around a diamond inheritance
        # which I cannot support with the current
        # super() call structure)
        #
        # Then pass this to the CoordTrans initialzer
        super().__init__(df,name = 'Composite coord. trans.',
                         Qpstr = B.Qpstr, Qstr = A.Qstr) 
                            
        self.A = A
        self.B = B 
        
    def diagram(self):
        
        diag = self.B.diagram()
        diag += self.A.diagram()
        
        return diag
    
    def __repr__(self):
        
        return f"CompositeCoordTrans({self.A!r}, {self.B!r})"
    
class QTransCoordSys(CoordSys):
    """ Input-transformed coordinate system """
    
    def __init__(self, T, C):
        """ X = C(T(Q')) """
        
        # First make a composite DFun 
        df = dfun.CompositeDFun(C,T)
        super().__init__(df._feval, nQ = T.nQp, nX = C.nX,
                         name = 'Trans. coord. sys.', 
                         Qstr = T.Qpstr, Xstr = C.Xstr , 
                         maxderiv = df.maxderiv, 
                         isatomic = C.isatomic, 
                         zlevel = df.zlevel)
        self.T = T # CoordTrans
        self.C = C # CoordSys
    
    def diagram(self):
        
        Tdiag = self.T.diagram() # The coordinate transformation
        Cdiag = self.C.diagram() # The untransformed CS
        
        Tdiag = Tdiag.replace("\n",
                              8 * " " + "│" + 8 * " " + "\n")
        
        Tdiag = Tdiag.replace("│↓               │",
                              "│↓              ↑│")
        return Tdiag + Cdiag 
    
    def __repr__(self):
        
        return f"QTransCoordSys({self.T!r},{self.C!r})"
                