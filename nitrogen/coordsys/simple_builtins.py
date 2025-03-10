from .coordsys import CoordSys, CoordTrans
import nitrogen.autodiff.forward as adf
import nitrogen.dfun as dfun
import numpy as np

__all__ = ['Valence3','CartesianN','LinearTrans','Polar','Cylindrical',
           'Spherical','PathTrans','TriatomicRadialPolar','Valence4']

class Valence3(CoordSys):
    """
    A triatomic valence coordinate system.
    
    The coordinates are :math:`r_1`, :math:`r_2`, and :math:`\\theta`.
    See Notes for embedding conventions.
    
    If `supplementary` then :math:`\\theta \\leftarrow \\pi - \\theta` is
    used.
    
    """
    
    def __init__(self, name = 'Triatomic valence', angle = 'rad', supplementary = False,
                 embedding_mode = 0):
        
        """
        Create a new Valence3 object.
        
        Parameters
        ----------
        name : str, optional
            The coordinate system name. The default is 'Triatomic valence'.
        angle : {'rad', 'deg'}, optional
            The degree units. The default is radians ('rad').
        supplementary : bool, optional
            If True, then the angle supplement is used. The default is False.
        embedding_mode : int, optional
            Select the frame embedding convention. The default is 0. See 
            Notes for details.
            
            
        Notes
        -----
        
        For `embedding_mode` = 0, the Cartesian coordinates are
        
        ..  math::
            
            X_0 &= (0, 0, -r_1) \\\\
            X_1 &= (0, 0, 0) \\\\
            X_2 &= (0, r_2  \\sin \\theta, -r_2 \\cos\\theta)
        
        For `embedding_mode` = 1, the Cartesian coordinates are
        
        ..  math::
            
            X_0 &= (r_1 \\cos \\theta/2, 0, r_1 \\sin \\theta/2) \\\\
            X_1 &= (0, 0, 0) \\\\
            X_2 &= (r_2 \\cos \\theta/2, 0, -r_2 \\sin \\theta/2)
        
        """
        
        super().__init__(self._csv3_q2x, nQ = 3, 
                         nX = 9, name = name, 
                         Qstr = ['r1', 'r2', 'theta'],
                         maxderiv = None, isatomic = True,
                         zlevel = None)
        
        if angle == 'rad' or angle == 'deg':
            self.angle = angle 
        else:
            raise ValueError('angle must be rad or deg')
            
        if not embedding_mode in [0,1]:
            raise ValueError('unexpected embedding_mode')
            
        self.supplementary = supplementary 
        self.embedding_mode = embedding_mode 
        
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
        
        if self.angle == 'deg':
            q[2] = (np.pi / 180.0) * q[2] 
        # q[2] is now in radians
        if self.supplementary:
            q[2] = np.pi - q[2] # theta <-- pi - theta 
        
        if self.embedding_mode == 0:
            np.copyto(out[:,2], (-q[0]).d ) # -r1
            np.copyto(out[:,7], (q[1] * adf.sin(q[2])).d ) #  r2 * sin(theta)
            np.copyto(out[:,8], (-q[1] * adf.cos(q[2])).d ) # -r2 * cos(theta)
        elif self.embedding_mode == 1:
            np.copyto(out[:,0], (q[0] * adf.cos(q[2]/2)).d ) # r1 * cos(theta/2)
            np.copyto(out[:,2], (q[0] * adf.sin(q[2]/2)).d ) # r1 * sin(theta/2)
            np.copyto(out[:,6], (q[1] * adf.cos(q[2]/2)).d ) # r2 * cos(theta/2)
            np.copyto(out[:,8], (-q[1] * adf.sin(q[2]/2)).d ) # -r2 * sin(theta/2)
        else:
            raise RuntimeError("Unexpected embedding_mode")
        
        return out

    def __repr__(self):
        return f"Valence3({self.name!r},{self.name!r})"
    
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
    
class Valence4(CoordSys):
    """
    A tetratomic valence coordinate system.
    
    The coordinates are :math:`r_1`, :math:`r_2`, :math:`r_3`,
    :math:`\\theta_1`, :math:`\\theta_2`, and :math:`\\phi`.
    See Notes for embedding conventions.
    
    """
    
    def __init__(self, name = 'Tetratomic valence', angle = 'rad', 
                 embedding_mode = 'C2'):
        
        """
        Create a new Valence3 object.
        
        Parameters
        ----------
        name : str, optional
            The coordinate system name. The default is 'Tetratomic valence'.
        angle : {'rad', 'deg'}, optional
            The degree units. The default is radians ('rad').
        embedding_mode : {'C2'}, optional
            The embedding mode, see Notes.
            
        Notes
        -----
        
        For :math:`r_1 = r_2` and :math:`\\theta_1 = \\theta_2`, the :math:`C_2`
        axis is parallel to :math:`z` for embedding mode ``'C2'``.
        
        """
        
        super().__init__(self._csv4_q2x, nQ = 6, 
                         nX = 12, name = name, 
                         Qstr = ['r1', 'r2', 'r3', 'theta1', 'theta2', 'phi'],
                         maxderiv = None, isatomic = True,
                         zlevel = None)
        
        if angle == 'rad' or angle == 'deg':
            self.angle = angle 
        else:
            raise ValueError('angle must be rad or deg')
            
        if not embedding_mode in ['C2']:
            raise ValueError('unexpected embedding_mode')
            
        # self.supplementary = supplementary 
        self.embedding_mode = embedding_mode 
        
    def _csv4_q2x(self, Q, deriv = 0, out = None, var = None):
        """
        Tetratomic valence coordinate system Q2X instance method.
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
        
        natoms = 4 
        base_shape =  Q.shape[1:]
        
        if var is None:
            var = [0, 1, 2, 3, 4, 5] # Calculate derivatives for all Q
        
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
        # q = r1, r2, r3, theta1, theta2, phi 
        
        if out is None:
            out = np.ndarray( (nd, 3*natoms) + base_shape, dtype = Q.dtype)
        out.fill(0) # Initialize out to 0
        
        # Calculate Cartesian coordinates
        
        if self.angle == 'deg':
            q[3] = (np.pi / 180.0) * q[3] 
            q[4] = (np.pi / 180.0) * q[4]
            q[5] = (np.pi / 180.0) * q[5]
    
        # q[3-5] now in radians
      
        
        r1,r2,r3,th1,th2,phi = q 
        if self.embedding_mode == 'C2':
            
            #  C2 axis along z (for r1 == r2, and th1 == th2)
            # 
            #      1      4         ^ z
            #     r1\    / r2       |
            #        2--3           +-->x
            #         r3
            #               
            #np.copyto(out[:,2], (-q[0]).d ) # -r1
            #np.copyto(out[:,7], (q[1] * adf.sin(q[2])).d ) #  r2 * sin(theta)
            #np.copyto(out[:,8], (-q[1] * adf.cos(q[2])).d ) # -r2 * cos(theta)
            
            np.copyto(out[:,3], -(r3.d)/2) # -r3/2 
            np.copyto(out[:,6],  (r3.d)/2) # +r3/2 
            
            dx1 = r1 * adf.cos(th1)
            dy1 = r1 * adf.sin(phi/2) * adf.sin(th1)
            dz1 = r1 * adf.cos(phi/2) * adf.sin(th1) 
            
            dx2 = -r2 * adf.cos(th2)
            dy2 = -r2 * adf.sin(phi/2) * adf.sin(th2)
            dz2 = r2 * adf.cos(phi/2) * adf.sin(th2) 
            
            np.copyto(out[:,0], dx1.d - (r3.d)/2) 
            np.copyto(out[:,1], dy1.d)
            np.copyto(out[:,2], dz1.d)
            
            np.copyto(out[:,9],  dx2.d + (r3.d)/2)
            np.copyto(out[:,10], dy2.d)
            np.copyto(out[:,11], dz2.d)
            
        else:
            raise RuntimeError("Unexpected embedding_mode")
        
        return out

    def __repr__(self):
        return f"Valence4({self.name!r},{self.name!r})"
    
    def diagram(self):
        # using U+250X box and U+219X arrows
        diag = ""
        
        diag += "     │↓              ↑│        \n"
        diag += "     │Q [3]      [9] X│        \n"
        diag += "   ╔═╪════════════════╪═╗      \n"
        diag += "   ║ │ ┌────────────┐ │ ║      \n"
        diag += "   ║ ╰─┤ 4-atom val ├─╯ ║      \n"
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
        if deriv >= 1:
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
    Linear coordinate transformation plus constant offset.
    
    The output coordinates Qi are defined as
    
        :math:`Q_i = T_{ij} Q'_j + t_i`
        
    Attributes
    ----------
    
    T : ndarray
        The transformation matrix.
    t : ndarray
        The offset vector.
        
    """
    
    def __init__(self, T, t = None, Qpstr = None, name = None):
        """
        Create a LinearTrans object.

        Parameters
        ----------
        T : ndarray
            The linear transformation matrix.
        t : ndarray, optional
            An offset vector. If None, it is ignored.
        Qpstr: list of str, optional
            Labels for the new coordinates.
        name: str, optional
            Coordinate transformation name.

        """
        if np.ndim(T) != 2:
            raise ValueError("T must be 2-dimensional")
        m,n = T.shape
        # m is the number of Q coordinates
        # n is the number of Q' coordinates
            
        super().__init__(self._lintrans, nQp = n, nQ = m, 
                         name = name,
                         Qpstr = Qpstr, maxderiv = None,
                         zlevel = 1)
            
        self.T = T.copy()   # Transformation matrix, copy
        
        if t is None:
            self.t = None 
        else:
            self.t = t.copy() # Offset vector.
            if len(self.t) != m:
                raise ValueError("The length of t must equal the first dimension"
                                 " of T")
        
    def _lintrans(self, Qp, deriv = 0, out = None, var = None):
        """
        Qp : ndarray
            Shape (self.nQp, ...)
        out : ndarray
            Shape (nd, self.nQ, ...)
        """
        
        base_shape =  Qp.shape[1:]
        N = self.nQp # = self.nX, number of inputs
        nQ = self.nQ
        
        if var is None:
            var = [i for i in range(N)] # Calculate derivatives for all Qp
    
        nvar = len(var)
        nd = dfun.nderiv(deriv, nvar)
        
        if out is None:
            out = np.ndarray( (nd, nQ) + base_shape, dtype = Qp.dtype)
        
        out.fill(0) # Initialize derivative array to 0
        
        # 0th derivatives = values
        # Q_i = T_ij * Q'_j + t_i
        np.copyto(out[0],  np.tensordot(self.T, Qp, axes = (1,0)) )
        if self.t is not None:
            for i in range(nQ):
                out[0,i] += self.t[i]
            
        # 1st derivatives
        if deriv >= 1:
            for i in range(nvar):
                # derivatives with respect to k = var[i]
                # This is just the k^th column of T 
                for j in range(nQ):
                    out[i+1, j:(j+1)].fill(self.T[j,var[i]])
                
            
        # All higher derivatives are zero
        # zlevel reflects this, which other functions can
        # check to maximize efficiency.
        
        return out
    
    def __repr__(self):
        return f'LinearTrans({self.T!r}, t = {self.t!r}, Qpstr = {self.Qpstr!r})'
    
    def diagram(self):
        """ CoordTrans diagram string """
        
        sQ =f"[{self.nQ:d}]"
        sQp =f"[{self.nQp:d}]"
        
        if self.name is None:
            label = "T@Q' + t"
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

class PathTrans(CoordTrans):
    """
    
    A generic reaction path coordinate transformation
    
    Attributes
    ----------
    path_fun : DFun
        The coordinate path function.
    disp_fun : DFun
        The path displacement vectors.
    
    """
    
    def __init__(self, path_fun, disp_fun):
        """
        Create a PathTrans coordinate transformation.

        Parameters
        ----------
        path_fun : (1,) -> (nQ,) DFun
            The reaction path function, :math:`P_i(s)`.
        disp_fun : (1,) -> (nQ * n) DFun
            The path displacement vectors, :math:`T_{ij}(s)`.

        Notes
        -----
        
        The `nQ` output coordinates :math:`Q_i` are the 
        same as those given by the path function object,
        `path_fun`. Its single argument, :math:`s`, 
        is the path parameter.
        
        An additional `n` coordinates :math:`d_j` are defined
        by the displacements vectors :math:`T_{ij}(s)`
        returned by `disp_fun`. (`disp_fun` returns the elements
        of `T` in row major order.)
        
        The input variable order is :math:`(s,d_0,d_1,\\ldots,d_{n-1})`.
        
        The output coordinates are calculated as 
        
        ..  math::
                
            Q_i = P_i(s) + \\sum_{j = 0}^{n-1} T_{ij}(s) d_j \\quad (i = 0\\ldots nQ-1)
        
        """
        
        if path_fun.nx != 1 or disp_fun.nx != 1:
            raise ValueError("path_fun and disp_fun must have 1 input variable")
        
        # The number of output coordinates is defined by the
        # number of output functions of `path_fun`
        nQ = path_fun.nf
        
        # The number of input coordinates equals the number
        # of path-displacements plus 1.
        if disp_fun.nf % nQ != 0:
            raise ValueError('dispfun.nf must be a multiple of nQ')
        n = disp_fun.nf // nQ # The number of displacement coordinates 
        
        nQp = n + 1 # The total number of input coordinates 
        
        Qpstr = ['s'] + [f"d{i:d}" for i in range(n)]
        Qstr = [f"Q{i:d}" for i in range(nQ)]
        
        # The maxderiv is the minimum of either path_fun 
        # or disp_fun 
        maxderiv = dfun._merged_maxderiv(path_fun.maxderiv, disp_fun.maxderiv)
        
        # If the displace vectors have a finite zlevel, then the 
        # zlevel for that term is just 1 more (from the linear dependence on 
        # Q'). The total zlevel is then the merged zlevel of the path function
        # and the displacement term 
        #
        if disp_fun.zlevel is None:
            disp_zlevel = None 
        else: 
            disp_zlevel = disp_fun.zlevel + 1 
        zlevel = dfun._merged_zlevel(path_fun.zlevel, disp_zlevel)
        
        super().__init__(self._pathtrans_fun, nQp = nQp, nQ = nQ,
                         name = 'Path', Qpstr = Qpstr, Qstr = Qstr,
                         maxderiv = maxderiv, zlevel = zlevel)
        
        self.path_fun = path_fun 
        self.disp_fun = disp_fun
        
    def _pathtrans_fun(self, Qp, deriv = 0, out = None, var = None):
        
        
        # Parse `out` and `var` first
        out,var = self._parse_out_var(Qp, deriv, out, var)
        nvar = len(var)
        # out ... (nd,nQ,...)
        
        # Calculate the derivatives of the Q with respect to 
        # the path parameter `s` and displacements `d`.
        # 
        #  Q_i = P_i(s) + T_ij(s) d_j
        #
        # We will consider two separate cases:
        # 1) If `s` is a requested variable, and 
        # 2) if `s` is not a requested variable.
        
        s = Qp[0:1] # (1,...)
        d = Qp[1:]  # (n,...)
        
        n = self.nQp - 1  # The number of displacements 
        base_shape = Qp.shape[1:]
        
        if 0 in var:
            #
            # `s` is requested. We need the derivatives
            # of P and T with respect to `s` up to the
            # requested derivative order
            #
            
            dP = self.path_fun.f(s, deriv = deriv) # (deriv+1, nQ, ...)
            dT = self.disp_fun.f(s, deriv = deriv) # (deriv+1, nQ*n,...)
            
            # Reshape dT to its array shape
            dT = np.reshape(dT, (deriv+1, self.nQ, n) + base_shape) 
            
            ################################
            # Now compute derivatives. Most of these will 
            # be zero. 
            out.fill(0.0)
            
            # Compute non-zero derivatives organized
            # by the total degree of displacement coordinates,
            # which has a maximum of 1.
            #
            s_pos = var.index(0) # The position of `s` in the multi-index
            m = nvar - s_pos - 1 # The number of variables after `s` in var order
            
            nck = adf.ncktab(nvar + deriv)
            
            for k in range(deriv + 1):
                
                if k <= deriv:
                    # Multi-index [0 ... k ... 0]
                    idx0 = nck[nvar + k, k] - nck[m + k, k]
                    np.copyto(out[idx0], dP[k] + np.einsum('ij...,j...->i...',dT[k],d))
                
                if (k+1) <= deriv:
                    
                    for j in range(0,s_pos):
                        # Consider multi-indices with a 1 before the k
                        #
                        #      j -p-   --m--
                        # [0.. 1 ... k ... 0 ]
                        # 
                        p = s_pos - j - 1 # The number of 0's between 1 and k
                        #
                        # The lexical increment from [0...k...0] to this 
                        #
                        delta_idx = nck[k+nvar,k+1] - nck[k+p+m+1,k+1]
                        idx1 = idx0 + delta_idx 
                        #
                        # The derivative is just the (k) derivative of 
                        #
                        # T_{i, var[j]-1}
                        #
                        np.copyto(out[idx1], dT[k,:,var[j]-1]) 
                    
                    for j in range(s_pos+1, nvar):
                        #
                        # Consider multi-indices with a 1 after the k 
                        #           ------m----
                        # [0 .... k ... 1 ... 0]
                        #           -p- j
                        #
                        # The lexical increment from [0...k...0] to this is
                        #
                        p_plus_1 = np.uint32(j - s_pos)
                        
                        delta_idx = nck[k+m,k] + nck[k+nvar,k+1] - nck[k+1+m,k+1] + p_plus_1
                        idx1 = idx0 + delta_idx 
                        np.copyto(out[idx1], dT[k,:,var[j]-1]) 
                #
                # Derivatives with total displacement degree
                # greater than or equal to 2 are strictly 
                # zero.
                #
                
        else:
            # `s` is not requested
            # only P and T values are needed
            # Reshape T to its array shape
            P = self.path_fun.val(s)  # (nQ,...)
            T = self.disp_fun.val(s)  # (nQ*n,...)
            T = np.reshape(T, (self.nQ,n) + base_shape) # T[i,j,...]
            
            # Calculate the zeroth derivative (i.e. value of Q_i)
            #
            # Qi = Pi + Tij dj
            #
            if deriv >= 0:
                np.copyto(out[0], P + np.einsum('ij...,j...->i...',T,d) )
            
            # Calculate the first derivatives. We know these are
            # all with respect to `d` variables because `s` is not
            # requested.
            # 
            # The derivative w.r.t. d_j is just T_ij
            #
            if deriv >= 1:
                for j in range(nvar):
                    np.copyto(out[j+1], T[:,var[j]-1])
            
            # All higher deriatives w.r.t d_j are zero 
            if deriv >=2 :
                out[(nvar+1):].fill(0.0) 
                
        return out 
        
    def __repr__(self):
        return f'PathTrans({self.path_fun!r}, {self.disp_fun!r})'
    
    def diagram(self):
        """ CoordTrans diagram string """
        
        sQ =f"[{self.nQ:d}]"
        sQp =f"[1+{self.nQp-1:d}]"
        
        diag = ""
        
        diag += "     │↓       \n"
        diag +=f"     │Q'{sQp:<6s}\n"
        diag += "   ╔═╧══════╗ \n"
        diag += "   ║  Path  ║ \n"
        diag += "   ║ Trans. ║ \n"
        diag += "   ║  Q(s)  ║ \n"
        diag += "   ╚═╤══════╝ \n"
        diag +=f"     │Q {sQ:<6s}\n"
        
        return diag
        
    
    
class Polar(CoordSys):
    """
    Polar coordinates :math:`(r,\\phi)` in two dimensions.
    
    .. math::
        
       x &= r \\cos\\phi 
       
       y &= r \\sin\\phi 
     
    Attributes
    ----------
    
    angle : {'deg','rad'}
        The angular unit.
    
    """
    
    def __init__(self, angle = 'deg'):
        
        """
        Create a new Polar coordinate system object.
        
        Parameters
        ----------
        angle : {'deg', 'rad'}, optional
            The angular unit, degrees or radians. The default is 'deg'.
        
        """
        
        name = "Polar"
        Qstr = ["r", "phi"]
        Xstr = ["x", "y"]
        
        super().__init__(self._csPolar_q2x, nQ = 2,
                         nX = 2, name = name, 
                         Qstr = Qstr, Xstr = Xstr,
                         maxderiv = None, isatomic = False,
                         zlevel = None)
        
        if angle == 'deg' or angle == 'rad':
            self.angle = angle  # 'deg' or 'rad'
        else:
            raise ValueError('angle must be ''deg'' or ''rad''.')
    
    def _csPolar_q2x(self, Q, deriv = 0, out = None, var = None):
        
        # Use adf routines to compute derivatives
        #
        q = dfun.X2adf(Q, deriv, var)
        # q[0] is r, q[1] = phi (in deg or rad)
        
        if self.angle == 'rad':
            x = q[0] * adf.cos(q[1])
            y = q[0] * adf.sin(q[1])
        else: # degree
            deg2rad = np.pi/180.0
            x = q[0] * adf.cos(deg2rad * q[1])
            y = q[0] * adf.sin(deg2rad * q[1])
        
        return dfun.adf2array([x,y], out) 
    
    def __repr__(self):
        return f'Polar(angle = {self.angle!r})'
    
    def diagram(self):
        
        # using U+250X box and U+219X arrows
        diag = ""
        
        sQ =f"[{self.nQ:d}]"
        sX =f"[{self.nX:d}]"
        
        diag += "     │↓              ↑│        \n"
        diag +=f"     │Q {sQ:<5s}  {sX:>5s} X│        \n"
        diag += "   ╔═╪════════════════╪═╗      \n"
        diag += "   ║ ╰────────────────╯ ║      \n"
        diag += "   ║  (r,phi) -> (x,y)  ║      \n"
        diag += "   ║       Polar        ║      \n"
        diag += "   ╚════════════════════╝      \n"
        
        return diag
    
class Cylindrical(CoordSys):
    """
    Cylindirical coordinates :math:`(r,\\phi, z)` in three dimensions.
    
    .. math::
        
       x &= r \\cos\\phi 
       
       y &= r \\sin\\phi 
     
    Attributes
    ----------
    
    angle : {'deg','rad'}
        The angular unit.
    
    """
    
    def __init__(self, angle = 'deg'):
        
        """
        Create a new Cylindrical coordinate system object.
        
        Parameters
        ----------
        angle : {'deg', 'rad'}, optional
            The angular unit, degrees or radians. The default is 'deg'.
        
        """
        
        name = "Cylindrical"
        Qstr = ["r", "phi", "z"]
        Xstr = ["x", "y", "z"]
        
        super().__init__(self._csCylindrical_q2x, nQ = 3,
                         nX = 3, name = name, 
                         Qstr = Qstr, Xstr = Xstr,
                         maxderiv = None, isatomic = False,
                         zlevel = None)
        
        if angle == 'deg' or angle == 'rad':
            self.angle = angle  # 'deg' or 'rad'
        else:
            raise ValueError('angle must be ''deg'' or ''rad''.')
    
    def _csCylindrical_q2x(self, Q, deriv = 0, out = None, var = None):
        
        # Use adf routines to compute derivatives
        #
        q = dfun.X2adf(Q, deriv, var)
        # q[0] is r, q[1] = phi (in deg or rad), q[2] = z
        
        if self.angle == 'rad':
            x = q[0] * adf.cos(q[1])
            y = q[0] * adf.sin(q[1])
        else: # degree
            deg2rad = np.pi/180.0
            x = q[0] * adf.cos(deg2rad * q[1])
            y = q[0] * adf.sin(deg2rad * q[1])
        # z = z
        z = q[2] 
        
        return dfun.adf2array([x,y,z], out) 
    
    def __repr__(self):
        return f'Cylindrical(angle = {self.angle!r})'
    
    def diagram(self):
        
        # using U+250X box and U+219X arrows
        diag = ""
        
        sQ =f"[{self.nQ:d}]"
        sX =f"[{self.nX:d}]"
        
        diag += "     │↓              ↑│        \n"
        diag +=f"     │Q {sQ:<5s}  {sX:>5s} X│        \n"
        diag += "   ╔═╪════════════════╪═╗      \n"
        diag += "   ║ ╰────────────────╯ ║      \n"
        diag += "   ║ (r,phi,z)->(x,y,z) ║      \n"
        diag += "   ║    Cylindrical     ║      \n"
        diag += "   ╚════════════════════╝      \n"
        
        return diag
    
class Spherical(CoordSys):
    """
    Spherical coordinates :math:`(r,\\theta,\\phi)` in three dimensions.
    
    .. math::
        
       x &= r \\sin\\theta\\cos\\phi 
       
       y &= r \\sin\\theta\\sin\\phi 
       
       z &= r \\cos\\theta
     
    Attributes
    ----------
    
    angle : {'deg','rad'}
        The angular unit.
    
    """
    
    def __init__(self, angle = 'deg'):
        
        """
        Create a new Spherical coordinate system object.
        
        Parameters
        ----------
        angle : {'deg', 'rad'}, optional
            The angular unit, degrees or radians. The default is 'deg'.
        
        """
        
        name = "Spherical"
        Qstr = ["r", "theta", "phi"]
        Xstr = ["x", "y", "z"]
        
        super().__init__(self._csSpherical_q2x, nQ = 3,
                         nX = 3, name = name, 
                         Qstr = Qstr, Xstr = Xstr,
                         maxderiv = None, isatomic = False,
                         zlevel = None)
        
        if angle == 'deg' or angle == 'rad':
            self.angle = angle  # 'deg' or 'rad'
        else:
            raise ValueError('angle must be ''deg'' or ''rad''.')
    
    def _csSpherical_q2x(self, Q, deriv = 0, out = None, var = None):
        
        # Use adf routines to compute derivatives
        #
        q = dfun.X2adf(Q, deriv, var)
        # q[0] is r, q[1] = theta, q[2] = phi
        
        
        
        if self.angle == 'rad':
            sintheta = adf.sin(q[1])
            costheta = adf.cos(q[1])
            sinphi = adf.sin(q[2])
            cosphi = adf.cos(q[2])
        else: # degree
            deg2rad = np.pi/180.0
            sintheta = adf.sin(deg2rad * q[1])
            costheta = adf.cos(deg2rad * q[1])
            sinphi = adf.sin(deg2rad * q[2])
            cosphi = adf.cos(deg2rad * q[2])
     
        r = q[0] 
        
        x = r * sintheta * cosphi 
        y = r * sintheta * sinphi 
        z = r * costheta
        
        return dfun.adf2array([x,y,z], out) 
    
    def __repr__(self):
        return f'Spherical(angle = {self.angle!r})'
    
    def diagram(self):
        
        # using U+250X box and U+219X arrows
        diag = ""
        
        sQ =f"[{self.nQ:d}]"
        sX =f"[{self.nX:d}]"
        
        diag += "     │↓              ↑│        \n"
        diag +=f"     │Q {sQ:<5s}  {sX:>5s} X│        \n"
        diag += "   ╔═╪════════════════╪═╗      \n"
        diag += "   ║ ╰────────────────╯ ║      \n"
        diag += "   ║(r,th,phi)->(x,y,z) ║      \n"
        diag += "   ║     Spherical      ║      \n"
        diag += "   ╚════════════════════╝      \n"
        
        return diag
    

class TriatomicRadialPolar(CoordSys):
    """
    Triatomic radial-polar coordinate system, :math:`(R,\\rho,\\phi)`.
    
    The three atoms lie in the :math:`xy` plane. Their
    coordinates are 
    
    ..  math::
        
        x_1 &= \\frac{R}{\\sqrt{3} }\\left( 1 - \\rho \\cos \\phi \\right)

        y_1 &= \\frac{R}{\\sqrt{3} } \\rho \\sin \\phi 
        
        
        x_2 &= \\frac{R}{2 \\sqrt{3} } \\left ( -1 + \\rho\\cos\\phi + \\sqrt{3} \\rho \\sin \\phi \\right )
        
        y_2 &= \\frac{R}{2 \\sqrt{3} }  \\left( \\sqrt{3} + \\sqrt{3} \\rho\\cos\\phi - \\rho\\sin\\phi \\right )
        
        
        x_3 &= \\frac{R}{2 \\sqrt{3} }  \\left(-1 + \\rho\\cos\\phi   - \\sqrt{3} \\rho\\sin\\phi \\right )
        
        y_3 &= \\frac{R}{2 \\sqrt{3} }  \\left(-\\sqrt{3} - \\sqrt{3} \\rho\\cos\\phi - \\rho\\sin\\phi \\right )
        
        z_1 &= z_2 = z_3 = 0
    
    :math:`R > 0` controls the total size scale, :math:` 0 \\leq \\rho \\leq 1` is the deformation magnitude,
    and :math:`0 \\leq \\phi < 2 \\pi` determines the direction of deformation. For :math:`\\rho = 0`, the 
    three particles form a triangle with side length :math:`R`. For :math:`\\rho = 1`, the three
    particles are co-linear.
    
    Some useful identities include
    
    ..  math::
        
        \\frac{r_1^2 + r_2^2 + r_3^2}{3} &= R^2 ( 1 + \\rho^2 ) 
        
        \\frac{2r_1^2 - r_2^2 - r_3^2}{6} &=  R^2 \\rho \\cos \\phi 
        
        \\frac{r_2^2 - r_3^2}{\\sqrt{12}} &= R^2 \\rho \\sin \\phi
    
    where :math:`r_1 = | \\vec{x}_2 - \\vec{x}_3 |`, etc.
    """
    
    def __init__(self, name = 'Triatomic radial-polar', angle = 'rad'):
        
        """
        Parameters
        ----------
        name : str, optional
            The coordinate system name. The default is 'Triatomic radial-polar'.
        angle : {'rad', 'deg'}, optional
            The angular units. The default is radians.
    
        """
        
        super().__init__(self._tripolar_q2x, nQ = 3, 
                         nX = 9, name = name, 
                         Qstr = ['R', 'rho', 'phi'],
                         maxderiv = None, isatomic = True,
                         zlevel = None)
        
        if angle == 'rad' or angle == 'deg':
            self.angle = angle 
        else:
            raise ValueError('angle must be rad or deg')
            
        
    def _tripolar_q2x(self, Q, deriv = 0, out = None, var = None):
        """
        """
        
        
        nd, nvar = dfun.ndnvar(deriv, var, self.nx)
        q = dfun.X2adf(Q, deriv, var)
        
        out, var = self._parse_out_var(Q, deriv, out, var)
        
        R = q[0]
        rho = q[1] 
        phi = q[2]
        
        if self.angle == 'deg':
            phi = phi * (np.pi/180.0) 
        # phi is now in radians 
        
        # Calculate rho * cos(phi) and 
        # rho * sin(phi) 
        #
        cos = rho * adf.cos(phi)
        sin = rho * adf.sin(phi) 
        
        rt3 = np.sqrt(3.0) 
        
        Rp = R / rt3 
        
        x0 = Rp * (1.0 - cos)
        y0 = Rp * sin 
        
        x1 = Rp * ( -0.5    + 0.5 * cos + rt3/2 * sin )
        y1 = Rp * ( rt3/2 + rt3/2 * cos   - 0.5 * sin )
        
        x2 = Rp * (-0.5     + 0.5 * cos   - rt3/2 * sin )
        y2 = Rp * (-rt3/2 - rt3/2 * cos      -0.5 * sin )
        
        z0 = adf.const_like(0.0, R)
        z1 = adf.const_like(0.0, R)
        z2 = adf.const_like(0.0, R)
        
        return dfun.adf2array([x0,y0,z0,x1,y1,z1,x2,y2,z2], out)
       
    def __repr__(self):
        return f"TriatomicRadialPolar({self.name!r},{self.angle!r})"
    
    def diagram(self):
        # using U+250X box and U+219X arrows
        diag = ""
        
        diag += "     │↓              ↑│        \n"
        diag += "     │Q [3]      [9] X│        \n"
        diag += "   ╔═╪════════════════╪═╗      \n"
        diag += "   ║ │ ┌────────────┐ │ ║      \n"
        diag += "   ║ ╰─┤  X3 polar  ├─╯ ║      \n"
        diag += "   ║   └────────────┘   ║      \n"
        diag += "   ╚════════════════════╝      \n"
        
        return diag
    
    @staticmethod
    def DistanceSquared2RRhoPhi(rr1,rr2,rr3):
        """
        Calculate triatomic radial-polar coordinates
        from the squares of the 
        three internuclear distances
        
        Parameters
        ----------
        rr1,rr2,rr3 : array_like
            The squared internuclear distance
            
            
        Returns
        -------
        R, rho, phi : ndarray
            The coordinates. :math:`\\phi` is returned 
            in radians in the range :math:`[0,2\\pi)`.
            
        """
        
        # Calculate the square-distances of 
        # each pair of atoms.
        rr1 = np.array(rr1)
        rr2 = np.array(rr2) 
        rr3 = np.array(rr3)
        
        B = (rr1 + rr2 + rr3) / 3 # B = R**2 (1 + rho**2)
        
        x = (2*rr1 - rr2 - rr3) / 6   # x = R**2 rho cos
        y = (rr2 - rr3) / np.sqrt(12) # y = R**2 rho sin
            
        A2 = x**2 + y**2 # A^2,  A = R**2 rho
        
        phi = np.arctan2(y,x) # tan = y/x
        
        phi = np.mod(phi, 2*np.pi)  # Move [-pi,pi] to [0, 2*pi)
        
        R2 = (B/2) * (1 + np.sqrt(abs(1 - 4*A2/B**2))) # R**2
        R = np.sqrt(R2) 
        rho = np.sqrt(A2) / R2 
        
        return R, rho, phi
    
    @staticmethod
    def X2RRhoPhi(X): 
        """
        Calculate triatomic radial-polar coordinates
        from Cartesian coordinates.
        
        Parameters
        ----------
        X : (9,...) array_like
            The Cartesian coordinates
            
        Returns
        -------
        R, rho, phi : ndarray
            The coordinates. :math:`\\phi` is returned 
            in radians in the range :math:`[0,2\\pi)`.
            
        """
        
        X = np.array(X) 
        
        # Calculate the square-distances of 
        # each pair of atoms.
        rr1 = np.sum( (X[3:6] - X[6:9])**2, axis = 0)
        rr2 = np.sum( (X[0:3] - X[6:9])**2, axis = 0)
        rr3 = np.sum( (X[0:3] - X[3:6])**2, axis = 0)
        
        return TriatomicRadialPolar.DistanceSquared2RRhoPhi(rr1,rr2,rr3)
    
    @staticmethod 
    def Distance2RRhoPhi(r1,r2,r3):
        """
        Calculate triatomic radial-polar coordinates
        from the three internuclear distances
        
        Parameters
        ----------
        r1,r2,r3 : array_like
            The internuclear distance
            
            
        Returns
        -------
        R, rho, phi : ndarray
            The coordinates. :math:`\\phi` is returned 
            in radians in the range :math:`[0,2\\pi)`.
            
        """
        
        # Calculate the square-distances of 
        # each pair of atoms.
        r1 = np.array(r1)
        r2 = np.array(r2) 
        r3 = np.array(r3)
        
        rr1 = r1*r1
        rr2 = r2*r2
        rr3 = r3*r3
        
        return TriatomicRadialPolar.DistanceSquared2RRhoPhi(rr1,rr2,rr3)
    
    
        