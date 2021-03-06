from .coordsys import CoordSys, CoordTrans
import nitrogen.autodiff.forward as adf
import nitrogen.dfun as dfun
import numpy as np

__all__ = ['Valence3','CartesianN','LinearTrans','Polar','Cylindrical',
           'Spherical']

class Valence3(CoordSys):
    """
    A triatomic valence coordinate system.
    
    The coordinates are :math:`r_1`, :math:`r_2`, and :math:`\\theta`.
    The first atom is at :math:`(0, 0, -r_1)`.
    The second atom is at the origin.
    The third atom is at :math:`(0, r_2 \sin\\theta, -r_2 \cos\\theta )`.
    
    If `supplementary` then :math:`\\theta \\leftarrow \\pi - \\theta` is
    used.
    
    """
    
    def __init__(self, name = 'Triatomic valence', angle = 'rad', supplementary = False):
        
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
            
        self.supplementary = supplementary 
        
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
        
        np.copyto(out[:,2], (-q[0]).d ) # -r1
        np.copyto(out[:,7], (q[1] * adf.sin(q[2])).d ) #  r2 * sin(theta)
        np.copyto(out[:,8], (-q[1] * adf.cos(q[2])).d ) # -r2 * cos(theta)
        
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
            A square matrix.
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
        if m != n:
            raise ValueError("T must be square")
            
        super().__init__(self._lintrans, nQp = m, nQ = m, 
                         name = name,
                         Qpstr = Qpstr, maxderiv = None,
                         zlevel = 1)
            
        self.T = T.copy()   # Transformation matrix, copy
        
        if t is None:
            self.t = None 
        else:
            self.t = t.copy() # Offset vector.
        
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
        # Q_i = T_ij * Q'_j + t_i
        np.copyto(out[0],  np.tensordot(self.T, Qp, axes = (1,0)) )
        if self.t is not None:
            for i in range(N):
                out[0,i] += self.t[i]
            
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