# -*- coding: utf-8 -*-
"""
frames.py

Body-frame rotation and transformation
"""

from .coordsys import CoordSys, CoordTrans
import numpy as np 
import nitrogen.autodiff.forward as adf 
import nitrogen.dfun 
import nitrogen.angmom 


__all__ = ['PermutedAxisCoordSys', 'RotatedCoordSys', 'MovingFrameCoordSys',
           'calcRASangle','calcRASseries','SingleAxisR3DFun',
           'EckartCoordSys','MovingEckartCoordSys']


class PermutedAxisCoordSys(CoordSys):
    """
    Permute the axis ordering of a coordinate system.
    
    Attributes
    ----------
    new_axis_order : tuple
        The new axis order. The index values are the
        axes of the original coordinate system.
    cs : CoordSys
        The original coordinate system
        
    """
    
    def __init__(self, cs, new_axis_order):
        
        """
        Create a PermutedAxisCoordSys.
        
        Parameters
        ----------
        cs : CoordSys
            The original coordinate system.
        new_axis_order: array_like
            The new axis order in terms of the original axis indices. This
            is a 3-tuple. The coordinate value along the new axis `i` equals
            the coordinate value of the old axis `new_axis_order[i]`.
            Examples: (0,1,2) leaves the axis order
            unchanged. (1,0,2) permutes the first two axes. (1,2,0) makes
            the original y coordinates the new x coordinates, 
            the original z coordinates the new y coordinates,
            and the original x coordinates the new z coordinates.
        
        """
        
        if not cs.isatomic:
            raise ValueError("PermutedAxisCoordSys requires an atomic coord. sys.")
        
        # Create new xyz labels
        temp = [["x'{0}".format(i),"y'{0}".format(i),"z'{0}".format(i)] for i in range(cs.natoms) ]
        temp = [val for sub in temp for val in sub]
        Xstr = temp # ['x0', 'y0', 'z0', 'x1', 'y1', 'z1', ...]

        # Check for a valid new_axis_order
        # This must be a permutation of [0,1,2]
        if len(new_axis_order) != 3:
            raise ValueError("new_axis_order must be a 3-tuple")
        if np.any(np.sort(new_axis_order) != [0,1,2]):
            raise ValueError("new_axis_order must be a permutation of [0,1,2]")
            
        
        
        super().__init__(self._csPermAxis_q2x, 
                         nQ = cs.nQ, nX = cs.nX, name = 'Axis permutation',
                         Qstr = cs.Qstr, Xstr = Xstr, 
                         maxderiv = cs.maxderiv, isatomic = cs.isatomic,
                         zlevel = cs.zlevel)
        
        self.cs = cs 
        self.new_axis_order = tuple(new_axis_order)
        
        
    def _csPermAxis_q2x(self, Q, deriv = 0, out = None, var = None):
  
        # Permuted axis Q2X function
        #
        # First, calculate the X derivative array
        # in the original coordinate system
        # 
        # Use the same deriv, out and var
        #
        X = self.cs.Q2X(Q, deriv = deriv, out = out, var = var)
        #
        # X has shape (nd, natoms*3, ...)
        
        # Now permute the data in X 
        # 
        # I will do this by explicit re-writing of
        # data with a temp vector 
        
        temp = np.empty_like(X[:,0:3]) # Temp for a single atom
        
        for i in range(self.natoms): # Atom i
            
            np.copyto(temp, X[:, (3*i):(3*i+3)])
            
            np.copyto(X[:,3*i + 0], temp[:,self.new_axis_order[0]]) # Copy coordinate for new axis 0
            np.copyto(X[:,3*i + 1], temp[:,self.new_axis_order[1]]) # Copy coordinate for new axis 1
            np.copyto(X[:,3*i + 2], temp[:,self.new_axis_order[2]]) # Copy coordinate for new axis 2
        
        return X 
        
    def diagram(self):
        
        newxyz = ""
        for i in range(3):
            newxyz += "xyz"[self.new_axis_order[i]]
        
        diag = ""
        diag += "     │↓              ↑│        \n"
        diag += "     │            ╔═══╧═══╗    \n"
        diag += "     │            ║ xyz → ║    \n"
        diag +=f"     │            ║  {newxyz:3s}  ║    \n"
        diag += "     │            ╚═══╤═══╝    \n"
        
        Cdiag = self.cs.diagram() # The untransformed CS
        
        return diag + Cdiag 
    
    def __repr__(self):
        
        return f"PermutedAxisCoordSys({self.cs!r},{self.new_axis_order!r})"

class RotatedCoordSys(CoordSys):
    """
    A rotated axis coordinate system. The new coordinates :math:`x'` are
    related to the original coordinates :math:`x` by
    
    ..  math:: 
        
        \\vec{x}'_i = \\mathbf{R} \\vec{x}_i
    
    for each atom :math:`i`.
    
    Attributes
    ----------
    R : (3,3) ndarray
        The rotation matrix.
    cs : CoordSys
        The original coordinate system
        
    """
    
    def __init__(self, cs, R):
        
        """
        Create a RotatedCoordSys
        
        Parameters
        ----------
        cs : CoordSys
            The original coordinate system.
        R : (3,3) array_like
            The rotation matrix 
        
        """
        
        if not cs.isatomic:
            raise ValueError("RotatedCoordSys requires an atomic coord. sys.")
        
        # Create new xyz labels
        temp = [["x'{0}".format(i),"y'{0}".format(i),"z'{0}".format(i)] for i in range(cs.natoms) ]
        temp = [val for sub in temp for val in sub]
        Xstr = temp # ['x0', 'y0', 'z0', 'x1', 'y1', 'z1', ...]

        # Check for valid R 
        R = np.array(R)
        if R.shape != (3,3):
            raise ValueError("R must be a 3 x 3 matrix")
            
        
        
        super().__init__(self._csRotAxis_q2x, 
                         nQ = cs.nQ, nX = cs.nX, name = 'Rotated axes',
                         Qstr = cs.Qstr, Xstr = Xstr, 
                         maxderiv = cs.maxderiv, isatomic = cs.isatomic,
                         zlevel = cs.zlevel)
        
        self.cs = cs 
        self.R = R
        
        
    def _csRotAxis_q2x(self, Q, deriv = 0, out = None, var = None):
  
        # Rotated axis Q2X function
        #
        # First, calculate the X derivative array
        # in the original coordinate system
        # 
        # Use the same deriv, out and var
        #
        X = self.cs.Q2X(Q, deriv = deriv, out = out, var = var)
        #
        # X has shape (nd, natoms*3, ...)
        
        # Now permute the data in X 
        # atom by atom 
        
        temp = np.empty_like(X[:,0:3])     # Temp for a single atom
        
        for i in range(self.natoms): # Atom i
            
            Xorig = X[:, (3*i):(3*i+3)] # View of original coordinates (nd, 3, ...)
            
            # Calculate rotated coordinates
            np.einsum('ij,kj...->ki...',self.R, Xorig, out = temp)
            
            # Move new coordinates into X
            np.copyto(X[:,(3*i):(3*i+3)], temp)
        
        # X now contains the rotated coordinates for every atom 
        
        return X 
        
    def diagram(self):
        
        diag = ""
        diag += "     │↓              ↑│        \n"
        diag += "     │            ╔═══╧═══╗    \n"
        diag += "     │            ║ xyz → ║    \n"
        diag += "     │            ║ R(xyz)║    \n"
        diag += "     │            ╚═══╤═══╝    \n"
        
        Cdiag = self.cs.diagram() # The untransformed CS
        
        return diag + Cdiag 
    
    def __repr__(self):
        
        return f"RotatedCoordSys({self.cs!r},{self.R!r})"

class MovingFrameCoordSys(CoordSys):
    """
    A coordinate system rotated to a moving frame. 
    The new coordinates :math:`x'` are
    related to the original coordinates :math:`x` by
    
    ..  math:: 
        
        \\vec{x}'_i(q) = \\mathbf{R}(q) \\vec{x}_i(q)
    
    for each atom :math:`i`. :math:`\\mathbf{R}(q)` is a 
    coordinate dependent rotation matrix.
    
    Attributes
    ----------
    R : DFun
        The rotation matrix function.
    cs : CoordSys
        The original coordinate system
        
    """
    
    def __init__(self, cs, R):
        
        """
        Create a MovingFrameCoordSys
        
        Parameters
        ----------
        cs : CoordSys
            The original coordinate system.
        R : DFun
            The rotation matrix. Each element is provided by the
            DFun in *row major* order.
        
        """
        
        if not cs.isatomic:
            raise ValueError("RotatedCoordSys requires an atomic coord. sys.")
        
        # Create new xyz labels
        temp = [["x'{0}".format(i),"y'{0}".format(i),"z'{0}".format(i)] for i in range(cs.natoms) ]
        Xstr = [val for sub in temp for val in sub]

        # Check for valid R 
        if R.nf != 9:
            raise ValueError("R must be a 3 x 3 = 9 function DFun")
         
        #
        # The maxderiv is the smaller of the two maxderivs
        # The zlevel is the sum of the two zlevels
        #
        maxderiv = nitrogen.dfun._merged_maxderiv(cs.maxderiv, R.maxderiv)
        zlevel = nitrogen.dfun._sum_None(cs.zlevel, R.zlevel)
        
        super().__init__(self._csMovingFrame_q2x, 
                         nQ = cs.nQ, nX = cs.nX, name = 'Rotated axes',
                         Qstr = cs.Qstr, Xstr = Xstr, 
                         maxderiv = maxderiv, isatomic = cs.isatomic,
                         zlevel = zlevel)
        
        self.cs = cs 
        self.R = R
        
        
    def _csMovingFrame_q2x(self, Q, deriv = 0, out = None, var = None):
  
        # Moving frame rotated coord. sys. Q2X function
        #
        # First, calculate the X derivative array
        # in the original coordinate system
        # 
        # Use the same deriv, out and var
        #
        X = self.cs.Q2X(Q, deriv = deriv, out = out, var = var)
        #
        
        if var is None:
            nvar = self.nQ
        else:
            nvar = len(var)
            
            
        # Now calculate the rotation matrix as a function of Q 
        # Use the same deriv and var as for Q2X
        #
        R = self.R.f(Q, deriv = deriv, var = var)
        
        # X has shape (nd, natoms*3, ...)
        # R has shape (nd, 9, ...)
        
        nd = R.shape[0]             # Number of derivatives
        base_shape = R.shape[2:]    # Base shape 
        
        # We need to matrix-multiply R
        # times each atomic vector
        
        # We will do this using adf's built-in matrix
        # multiply 
        
        # First, create an adarray for R and re-shape
        # it as necessary 
        R = np.reshape(R, (nd, 3, 3) + base_shape ) # (nd, 3, 3, ...)
        R_ad = adf.array(R, deriv, nvar, zlevel = self.R.zlevel) # (no copy)
        R_ad = R_ad.moveaxis_base([0,1],[-2,-1]) # Move matrix indices to end
        # R_ad's base_shape is now (...,3,3)
        
        
        # To save some temp memory, we will
        # rotate atom-by-atom
        
        for i in range(self.natoms): # Atom i
            
            Xi = X[:,(3*i):(3*i+3)].reshape( (nd, 3, 1) + base_shape) # (nd, 3, 1, ...) 
            Xi_ad = adf.array(Xi, deriv, nvar, zlevel = self.cs.zlevel) # (no copy)
            Xi_ad = Xi_ad.moveaxis_base([0,1], [-2,-1]) # Move matrix indices to end 
            # Xi_ad's base_shape is now (..., 3, 1)
            
            new_Xi = R_ad @ Xi_ad # Matrix multiply on final pairs of indices
            # new_Xi has base_shape (..., 3, 1)
            new_Xi = new_Xi.moveaxis_base([-2,-1], [0,1]) # Now (3, 1, ...)
            new_Xi = new_Xi.reshape_base((3,) + base_shape)
            
            # Finally, copy new coordinates to output 
            np.copyto(X[:,(3*i):(3*i+3)], new_Xi.d)
        
        # X now contains the rotated coordinates for every atom 
        return X 
        
    def diagram(self):
        
        diag = ""
        diag += "     │↓              ↑│        \n"
        diag += "     │           ╔════╧════╗   \n"
        diag += "     │     →     ║  xyz →  ║   \n"
        diag += "     ├───────────╢R(Q)(xyz)║   \n"
        diag += "     │           ╚════╤════╝   \n"
        
        Cdiag = self.cs.diagram() # The untransformed CS
        
        return diag + Cdiag 
    
    def __repr__(self):
        
        return f"RotatedCoordSys({self.cs!r},{self.R!r})"
    

def _R3axis(theta, axis):
    
    # Rotation matrix about a single 
    # axis: 0,1,2 = x,y,z
    # theta in rad
    
    c = np.cos(theta)
    s = np.sin(theta)
    if axis == 0:
        R = np.array([
            [1., 0., 0.],
            [0., c, -s],
            [0., +s, c]])
    elif axis == 1:
        R = np.array([
            [c, 0., +s],
            [0., 1., 0.],
            [-s,  0., c]])
    elif axis == 2:
        R = np.array([
            [c, -s, 0.],
            [+s, c, 0.],
            [0., 0., 1.]])
    else:
        raise ValueError("invalid axis")
    return R 

    
def calcRASangle(cs, mass, Qref, Qstar_idx, Qstar_final_value, axis,
                 int_points = 100, Rref = None):
    """
    Calculate the reduced axis system (RAS)
    for an aperiodic coordinate :math:`Q^*` with a
    fixed rotation axis via numerical integration.

    Parameters
    ----------
    cs : CoordSys
        The coordinate system
    mass : array_like
        The atomic masses 
    Qref : array_like
        The reference geometry.
    Qstar_idx : integer
        The coordinate index of :math:`Q^*`
    Qstar_final_value : scalar
        The integration end-point of :math:`Q^*`.
    axis : integer
        The RAS rotation axis after rotation to the 
        intermediate reference frame via `Rref`. This 
        must be 0, 1, or 2 (for the first, second, or
        third axis).
    int_points : integer, optional
        The number of integration steps. The default is 100. 
    Rref : (3,3) array_like, optional
        The fixed rotation matrix from the original coordinate 
        system frame to an intermediate frame. If None (default),
        then the inertial principal axis system will be used. 
        In this case, `axis` = 0, 1, or 2 specifies the :math:`a`,
        :math:`b`, or :math:`c` principal axis, respectively.

    Returns
    -------
    Rref : (3,3) ndarray
        The rotation axis to the intermediate reference system.
        If `Rref` is None, then this is the PAS of the reference geometry.
    Qstar_grid : ndarray
        The :math:`Q^*` integration grid.
    theta : ndarray
        The integrated value of the rotation angle :math:`\\theta` 
        (in radians) on the :math:`Q^*` grid.
        
    
    Notes
    -----
    The RAS is defined in [Picket1972]_. This function assumes the 
    special case of a fixed direction of rotation along the large
    amplitude motion (LAM) coordinate :math:`Q^*`, e.g. an axis
    of common symmetry or the direction normal to a plane of symmetry.
    
    The reference coordinate system is evaluated at the supplied 
    coordinates :math:`Q_\\mathrm{ref}` and, by default, moved to its inertial
    principal axis system (PAS). A different choice of reference can 
    be passed via `Rref`. The rotation matrix which rotates the 
    coordinates from the original frame to the intermediate frame is 
    returned as :math:`\\mathbf{R}_\\mathrm{ref}`.
    
    The RAS usually coincides with the reference PAS at the reference geometry.
    As :math:`Q^*` is displaced from its reference value, the RAS frame 
    is rotated relative to the reference frame by an angle :math:`\\theta(Q^*)` 
    about the axis specified by `axis`. The coordinates in the 
    final RAS are thus
    
    ..  math::
        
        \\vec{x}_\\mathrm{RAS} = \\mathbf{R}(\\theta(Q^*)) \\mathbf{R}_\\mathrm{ref} \\vec{x},
    
    where :math:`\\vec{x}` refers to the original coordinate system `cs`.
    :math:`\\mathbf{R}_\\mathrm{ref}` orders the principal axes as :math:`(a,b,c)`
    by default. If 
    `axis` is ``0``, then 
    
    ..  math::
        
        \\mathbf{R}(\\theta) = \\left( \\begin{array}{ccc} 1 & 0 & 0 \\\\
                                      0 & \\cos\\theta & -\\sin\\theta \\\\ 
                                      0 & \\sin\\theta & \\cos\\theta \\end{array} \\right)
            
    and cyclic permutations for ``1`` and ``2``. Special care should be taken
    for degenerate inertial axes. 
    
    See Also
    --------
    calcRASseries : Calculate the RAS angle power series. 
    
    References
    ----------
    .. [Picket1972] H. M. Pickett, "Vibration-Rotation Interactions and the Choice of 
       Rotating Axes for Polyatomic Molecules," J. Chem. Phys., 56, 1715 (1972).
       https://doi.org/10.1063/1.1677430
     
    
    """
    
    #################
    # Parse rotation axis
    #
    rot_axis = axis 
    if rot_axis not in [0,1,2]:
        raise ValueError("Unexpected axis (0, 1, or 2)")
    
    ##############
    # Evaluate the initial coordinate system
    # at the reference geometry 
    #
    X0 = cs.Q2X(Qref)[0] # (N*3,) Cartesian coordinates
    #
    #
    if Rref is None:
        # Calculate the principal axis system of the 
        # reference configuration
        #
        _,RPAS,_  = nitrogen.angmom.X2PAS(X0, mass)
        Rref = RPAS 
    else:
        Rref = np.array(Rref)
        if Rref.shape != (3,3):
            raise ValueError("Rref must be a 3 x 3 matrix")

    N = len(X0)//3  # The atom count
    
    # Create a uniform integration grid
    q_initial = Qref[Qstar_idx]
    q_final = Qstar_final_value 
    q_grid = np.linspace(q_initial, q_final, int_points) # The integration grid
    dq = q_grid[1] - q_grid[0] # The step size 
    q_grid_half = q_grid + dq/2 # The half-step grid 
    
    
    # Calculate the coordinates and derivatives in the 
    # reference frame
    mass = np.array(mass) # The masses 
    
    Qgrid = np.tile(Qref, (int_points,1)).T
    Qgrid[Qstar_idx,:] = q_grid
    Xref = cs.Q2X(Qgrid, deriv = 1, var = [Qstar_idx]) # (nd=2, N*3, int_points)
    Xref = np.reshape(Xref, (2, N, 3, int_points))     # (2, N, 3, int_points)
    Xcom = np.sum(Xref * mass.reshape((1,N,1,1)), axis = 1) / sum(mass)
    Xref = Xref - Xcom.reshape((2,1,3,int_points)) # Subtract COM
    Xref = np.einsum('ij,kljn->klin', Rref, Xref) # Rotate to reference frame
    
    # Do again for the half-step grid
    Qgrid_half = Qgrid = np.tile(Qref, (int_points,1)).T
    Qgrid_half[Qstar_idx,:] = q_grid_half
    Xref_half = cs.Q2X(Qgrid_half, deriv = 1, var = [Qstar_idx]) # (nd=2, N*3, int_points)
    Xref_half = np.reshape(Xref_half, (2, N, 3, int_points))     # (2, N, 3, int_points)
    Xcom_half = np.sum(Xref_half * mass.reshape((1,N,1,1)), axis = 1) / sum(mass)
    Xref_half = Xref_half - Xcom_half.reshape((2,1,3,int_points)) # Subtract COM
    Xref_half = np.einsum('ij,kljn->klin', Rref, Xref_half) # Rotate to reference frame
    
    
    def dtheta(Xq, theta, mass, axis):
        #
        # Xq : (nd,N,3), value + first derivative of coordinates in reference frame
        # theta : rotation angle, rad
        # mass : masses 
        
        R = _R3axis(theta, axis) 
        
        I = (axis + 1) % 3 # The axis after `axis` in cyclic right-hand order
        J = (axis + 2) % 3 # The second axis after `axis` in cyclic right-hand order
        
        
        Xt = np.tensordot(R, Xq, [[1],[2]]) # Result: (3,nd,N)
        
        b = Xt[I,0,:]
        c = Xt[J,0,:]
        db = Xt[I,1,:] # db/dq
        dc = Xt[J,1,:] # dc/dq
        
        
        dthetadq = sum(mass * (c * db - b * dc)) / sum( mass * (b**2 + c**2))
        
        return dthetadq
        
    # Calculate RK4 integral 
    theta_grid = np.zeros((int_points,))

    #
    # The initial value of theta is 0 radians.
    #
    for i in range(int_points - 1):
        #
        # There are 4 terms in the standard
        # RK4 expression
        #
        k1 = dtheta(Xref[...,i], theta_grid[i], mass, rot_axis) 
        k2 = dtheta(Xref_half[...,i], theta_grid[i] + dq*k1/2, mass, rot_axis) 
        k3 = dtheta(Xref_half[...,i], theta_grid[i] + dq*k2/2, mass, rot_axis) 
        k4 = dtheta(Xref[...,i+1], theta_grid[i] + dq*k3, mass, rot_axis) 
        
        # Add their weighted sum
        theta_grid[i+1] = theta_grid[i] + (dq/6) * (k1 + 2*k2 + 2*k3 + k4) 
    
    
    return Rref, q_grid, theta_grid 
    
    
def calcRASseries(cs, mass, Qref, Qstar_idx, degree, axis, Rref = None):
    
    """
    Calculate the reduced axis system (RAS)
    for an aperiodic coordinate :math:`Q^*` with a
    fixed rotation axis via a partial power series.

    Parameters
    ----------
    cs : CoordSys
        The coordinate system
    mass : array_like
        The atomic masses 
    Qref : array_like
        The reference geometry.
    Qstar_idx : integer
        The coordinate index of :math:`Q^*`
    degree : integer
        The maximum degree of the :math:`\\theta(Q^*)` power series
    axis : integer
        The rotation axis of the reference geometry. This must 
        be 0,1,2
    Rref : (3,3) array_like, optional
        The fixed rotation matrix from the original coordinate 
        system frame to an intermediate frame. If None (default),
        then the inertial principal axis system will be used. 
        In this case, `axis` = 0, 1, or 2 specifies the :math:`a`,
        :math:`b`, or :math:`c` principal axis, respectively.
        
    Returns
    -------
    Rref : (3,3) ndarray
        The rotation axis to the intermediate reference system
    pow : (degree+1,) ndarray
        The power series approximation of the rotation angle :math:`\\theta`
        (in radians) as a function of the displacement of :math:`Q^*` from 
        the reference value.
        
    
    Notes
    -----
    See the Notes to :func:`calcRASangle()`
    for more details.
    
    See Also
    --------
    calcRASangle : Calculate the RAS angle via numerical integration. 
      
    """
    
    ###########################
    # Calculate the RAS angle via a power series solution
    #
    
    #################
    # Parse rotation axis
    #
    rot_axis = axis 
    if rot_axis not in [0,1,2]:
        raise ValueError("Unexpected axis (0, 1, or 2)")
        
    pow_order = degree # Maximum power in theta(q) power series 
    nd = pow_order + 1 # The number of derivatives (including zeroth)
    N = len(mass)      # The number of atoms, N 
    
    # Calculate the coordinates and their derivatives w.r.t. Q*
    dX = cs.Q2X(Qref, deriv = pow_order, var = [Qstar_idx]) # (nd, N*3)
    
    if Rref is None:
        _,RPAS,_ = nitrogen.angmom.X2PAS(dX[0], mass) # Calculate PAS frame
        Rref = RPAS 
    else:
        Rref = np.array(Rref)
        if Rref.shape != (3,3):
            raise ValueError("Rref must be a 3 x 3 matrix")
            
    dX = np.reshape(dX, (nd, N, 3)) # (nd,N,3)
    
    
    # Subtract center-of-mass 
    Xcom = np.sum( dX * np.array(mass).reshape(1,N,1), axis = 1) / sum(mass) 
    dX = dX - Xcom.reshape((nd,1,3))
    # Rotate to reference frame
    dX0 = np.einsum('ij,klj->kli',Rref,dX) # (nd,N,3)
    # dX contains the coordinate derivatives in the reference frame
    #
    
    # Create adarray objects for exact derivative calculation
    
    I = (rot_axis + 1) % 3 # The axis after `axis` in cyclic right-hand order
    J = (rot_axis + 2) % 3 # The second axis after `axis` in cyclic right-hand order
    
    # We will calculate the adarray for (dtheta/dQ*), which therefore
    # only needs derivatives up to pow_order - 1 to provide derivatives
    # for theta(Q*) up to pow_order.
    #
    # 
    b = [adf.array(dX0[:-1,i,I], pow_order-1, 1) for i in range(N)]
    c = [adf.array(dX0[:-1,i,J], pow_order-1, 1) for i in range(N)]
    db = [ adf.array(adf.reduceOrder(dX0[:,i,I], 0, pow_order, 1, adf.idxtab(pow_order,1)), pow_order-1, 1) for i in range(N)]
    dc = [ adf.array(adf.reduceOrder(dX0[:,i,J], 0, pow_order, 1, adf.idxtab(pow_order,1)), pow_order-1, 1) for i in range(N)]
    
    num,den = 0,0
    for i in range(N):
        num = num + mass[i] * (c[i] * db[i] - b[i] * dc[i])
        den = den + mass[i] * (b[i] * b[i] + c[i] * c[i])
    dtheta = num / den 
    # dtheta is the derivative array of dtheta/dQ* with respect to Q*
    # Convert to coefficients a power series in Q*
    # The constant term is theta(Q*=0) = 0
    #
    theta_pow = np.zeros(pow_order + 1)
    for i in range(pow_order):
        
        theta_pow[i+1] = dtheta.d[i] / (i+1)
        # The factor of 1/(i+1) is necessary to 
        # convert from a power series of dtheta/dQ* to a power
        # series of theta.
    
    return Rref, theta_pow 

class SingleAxisR3DFun(nitrogen.dfun.DFun):
    """
    A diffentiable 3 x 3 rotation matrix for 
    rotation about a single axis with a 
    rotation angle power series w.r.t a single 
    variable.
    """
    
    def __init__(self, theta_pow, Q0, axis, ni, i):
        """
        Create a DFun object for the 3 x 3 rotation matrix
        about a given axis.
    
        Parameters
        ----------
        theta_pow : array_like
            The power series coefficients of the rotation angle :math:`\\theta`
            with respect to displacements of :math:`Q` from :math:`Q_0`.
        Q0 : scalar
            The reference value of :math:`Q`.
        axis : integer
            The axis index (0, 1, 2).
        ni : integer
            The total number of DFun variables.
        i : integer
            The variable index of :math:`Q` in the DFun.
    
    
        """
        
        super().__init__(self._singAxisR3, nf = 9, nx = ni)
        
        self.axis = axis 
        self.theta_pow = theta_pow.copy() 
        self.i = i 
        self.Q0 = Q0 
        
    def _singAxisR3(self, X, deriv = 0, out = None, var = None):
        
    
        nd,nvar = nitrogen.dfun.ndnvar(deriv, var, self.nx)
       
        if out is None:
            base_shape = X.shape[1:]
            out = np.ndarray( (nd, self.nf) + base_shape, dtype = X.dtype)
            
        out.fill(0.0) # Initialize to zero
        
        x = nitrogen.dfun.X2adf(X, deriv, var)
        Qi = x[self.i] # The coordinate that theta depends on 
        dQ = Qi - self.Q0 # The displacmenet of this coordinate from its reference value 
        
        theta_ad = adf.const_like(0.0, dQ)
        # theta_ad <--- 0.0 
        dQ_pow = adf.const_like(1.0, dQ) 
        # dQ_pow <-- 1.0 
        
        for i in range(len(self.theta_pow)):
            theta_ad = theta_ad + self.theta_pow[i] * dQ_pow 
            dQ_pow = dQ_pow * dQ 
        
        c = adf.cos(theta_ad) # cos(theta)
        s = adf.sin(theta_ad) # sin(theta) 
        
        if self.axis == 0:
            
            out[0:1,0].fill(1.0)        # R(0,0) <-- 1 
            np.copyto(out[:,4], c.d)    # R(1,1) <-- cos
            np.copyto(out[:,8], c.d)    # R(2,2) <-- cos
            np.copyto(out[:,5],-s.d)    # R(1,2) <-- -sin
            np.copyto(out[:,7],+s.d)    # R(2,1) <-- +sin 
            
        elif self.axis == 1:
            
            out[0:1,4].fill(1.0)        # R(1,1) <-- 1 
            np.copyto(out[:,0], c.d)    # R(0,0) <-- cos
            np.copyto(out[:,8], c.d)    # R(2,2) <-- cos
            np.copyto(out[:,6],-s.d)    # R(2,0) <-- -sin
            np.copyto(out[:,2],+s.d)    # R(0,2) <-- +sin 
            
        elif self.axis == 2:
            
            out[0:1,8].fill(1.0)        # R(2,2) <-- 1 
            np.copyto(out[:,0], c.d)    # R(0,0) <-- cos
            np.copyto(out[:,4], c.d)    # R(1,1) <-- cos
            np.copyto(out[:,1],-s.d)    # R(0,1) <-- -sin
            np.copyto(out[:,3],+s.d)    # R(1,0) <-- +sin 
            
        else:
            raise ValueError("Unexpected self.axis")
        
        
        return out 


def calcEckartC(X0, X, mass):
    """
    Calculate the mass-weighted distance 4 x 4
    quaternion matrix

    Parameters
    ----------
    X0 : (N*3,) array_like
        The reference geometry.
    X : (N*3,...) array_like
        The geometries for evaluating :math:`C`.
    mass : (N,) array_like
        The mass of each atom.

    Returns
    -------
    C : (...,4,4) ndarray
        The :math:`C` array for each geometry.

    """
    
    N = len(X0) // 3 # The number of atoms
    
    X = np.array(X).copy()
    base_shape = X.shape[1:] 
    
    mass = np.array(mass)
    
    # Subtract the center-of-mass 
    xcom = np.sum(X.reshape((N,3) + base_shape) * mass.reshape((N,1) + tuple([1]*len(base_shape))), axis = 0) / sum(mass)
    # xcom has shape (3,...)
    
    for i in range(N):
        for j in range(3):
            X[3*i + j] -= xcom[j] 
    # X is now the center-of-mass frame 
    
    # Make list of x,y,z coordinates 
    x = [X[3*i + 0] for i in range(N)]
    y = [X[3*i + 1] for i in range(N)]
    z = [X[3*i + 2] for i in range(N)]
    
    # Now calculate the Eckart rotation matrix
    # using the quaternion method
    #
    # Calculate sum and difference of Cartesians
    xp = [X0[3*i + 0] + x[i] for i in range(N)]
    yp = [X0[3*i + 1] + y[i] for i in range(N)]
    zp = [X0[3*i + 2] + z[i] for i in range(N)]
    
    xm = [X0[3*i + 0] - x[i] for i in range(N)]
    ym = [X0[3*i + 1] - y[i] for i in range(N)]
    zm = [X0[3*i + 2] - z[i] for i in range(N)]
    
    #
    # Evaluate C matrix elements
    # (Eq 24 of reference)
    #
    C = np.zeros(base_shape + (4,4))
    
    C[...,0,0] = sum([mass[i] * (xm[i]*xm[i] + ym[i]*ym[i] + zm[i]*zm[i]) for i in range(N)])
    C[...,1,1] = sum([mass[i] * (xm[i]*xm[i] + yp[i]*yp[i] + zp[i]*zp[i]) for i in range(N)])
    C[...,2,2] = sum([mass[i] * (xp[i]*xp[i] + ym[i]*ym[i] + zp[i]*zp[i]) for i in range(N)])
    C[...,3,3] = sum([mass[i] * (xp[i]*xp[i] + yp[i]*yp[i] + zm[i]*zm[i]) for i in range(N)])
    
    C12 = sum([mass[i] * (yp[i]*zm[i] - ym[i]*zp[i]) for i in range(N)])
    C13 = sum([mass[i] * (xm[i]*zp[i] - xp[i]*zm[i]) for i in range(N)])
    C14 = sum([mass[i] * (xp[i]*ym[i] - xm[i]*yp[i]) for i in range(N)])
    C23 = sum([mass[i] * (xm[i]*ym[i] - xp[i]*yp[i]) for i in range(N)])
    C24 = sum([mass[i] * (xm[i]*zm[i] - xp[i]*zp[i]) for i in range(N)])
    C34 = sum([mass[i] * (ym[i]*zm[i] - yp[i]*zp[i]) for i in range(N)])

    C[...,0,1] = C12 
    C[...,1,0] = C12
    
    C[...,0,2] = C13
    C[...,2,0] = C13
			   
    C[...,0,3] = C14
    C[...,3,0] = C14
			   
    C[...,1,2] = C23
    C[...,2,1] = C23
			   
    C[...,1,3] = C24
    C[...,3,1] = C24
			   
    C[...,2,3] = C34
    C[...,3,2] = C34
    
    return C

def _eckart_c2r(C):
    """
    Calculate Eckart rotation matrix from 
    C matrix 
    
    Parameters
    ----------
    C : (...,4,4) ndarray
        The C quaternion matrix.

    Returns
    -------
    R : (...,3,3) ndarray
        The rotation matrix.

    """
  
    #
    # Diagonalize C and get the derivatives
    # of the eigenvector with the lowest eigenvalue
    #
    _,V = np.linalg.eigh(C)
    v0 = V[...,:,0] # The lowest eigenvalue's eigenvector
    
    # Extract quaternion components 
    q0 = v0[...,0]
    q1 = v0[...,1]
    q2 = v0[...,2]
    q3 = v0[...,3]
    
    # Calculate rotation matrix 
    
    q00 = q0*q0
    q11 = q1*q1
    q22 = q2*q2
    q33 = q3*q3
    
    q01 = q0*q1
    q02 = q0*q2
    q03 = q0*q3
    q12 = q1*q2
    q13 = q1*q3
    q23 = q2*q3
    
    
    R11 = q00 + q11 - q22 - q33 
    R22 = q00 - q11 + q22 - q33 
    R33 = q00 - q11 - q22 + q33 
    
    R12 = 2*(q12 + q03)
    R13 = 2*(q13 - q02)
    R21 = 2*(q12 - q03)
    R23 = 2*(q23 + q01)
    R31 = 2*(q13 + q02)
    R32 = 2*(q23 - q01)
    
    R = np.array([
        [R11,R12,R13],
        [R21,R22,R23],
        [R31,R32,R33]
        ]) # (3,3,...)
    R = np.moveaxis(R, [0,1], [-2, -1])
    
    return R

def X2Eckart(X0, X, mass):
    """
    Calculate the Eckart frame configuration 
    and corresponding rotation matrix.

    Parameters
    ----------
    X0 : (N*3,) array_like
        The reference geometry.
    X : (N*3,...) array_like
        The geometries for evaluating :math:`C`.
    mass : (N,) array_like
        The mass of each atom.

    Returns
    -------
    Xeckart : (N*3,...) ndarray
        The configuration rotated into the the Eckart (and center-of-mass) frame
    R : (...,3,3) ndarray
        The rotation matrices rotating vectors from the original frame to the 
        Eckart frame.

    """
        
    N = len(X0) // 3 # The number of atoms
    
    X = np.array(X).copy()
    base_shape = X.shape[1:] 
    
    mass = np.array(mass)
    
    # Subtract the center-of-mass 
    xcom = np.sum(X.reshape((N,3) + base_shape) * mass.reshape((N,1) + tuple([1]*len(base_shape))), axis = 0) / sum(mass)
    # xcom has shape (3,...)
    
    for i in range(N):
        for j in range(3):
            X[3*i + j] -= xcom[j] 
    # X is now the center-of-mass frame 
    
    C = calcEckartC(X0, X, mass) # Calculate the 4 x 4 matrix 
            
    R = _eckart_c2r(C)  # Calculate the rotation matrix, (...,3,3)
    
    X = np.reshape(X, (N,3,) + base_shape ) # (N,3,...)
    XT = X.T # ( ...,3,N)
    Xeckart = (R @ XT).T # (N,3,...)
    Xeckart = np.reshape(Xeckart, (N*3,) + base_shape)
    
    return Xeckart, R 
    
    
class EckartCoordSys(CoordSys):
    """
    
    An Eckart frame coordinate system. This method
    uses the quaternion-based algorithm of [Kras2014soew]_.
    
    
    Attributes
    ----------
    cs : CoordSys
        The original coordinate system
    X0 : (3*N,) ndarray
        The reference Cartesian configuration
    mass : (N,) ndarray
        The atomic masses used to calculate the 
        Eckart frame
        
        
    References
    ----------
    
    References
    ----------
    .. [Kras2014soew] S. Krasnoshchekov, E. Isayeva, and N. Stepanov,
       "Determination of the Eckart molecule-fixed frame by use of the
       apparatus of quaternion algebra," J. Chem. Phys., 140, 154104 (2014).
       https://doi.org/10.1063/1.4870936
     
            
    """
    
    def __init__(self, cs, X0, mass):
        """
        Create an EckartCoordSys object.

        Parameters
        ----------
        cs : CoordSys
            The original coordinate system
        X0 : (3*N,) array_like
            The reference Cartesian configuration. The center-of-
            mass will be shifted to the origin.
        mass : (N,) array_like
            The mass of each atom. (This does not have to 
            equal the masses used for KEOs, but it should
            in practice for a useful Eckart frame.)

        """
        
        if not cs.isatomic:
            raise ValueError("Eckart frames must use atomic coordinate systems.")
            
        # Create new xyz labels
        temp = [["x'{0}".format(i),"y'{0}".format(i),"z'{0}".format(i)] for i in range(cs.natoms) ]
        Xstr = [val for sub in temp for val in sub]
            
        super().__init__(self._csEckartFrame_q2x,
                         nQ = cs.nQ, nX = cs.nX, name = 'Eckart frame',
                         Qstr = cs.Qstr, Xstr = Xstr, isatomic = cs.isatomic,
                         maxderiv = cs.maxderiv, zlevel = None)
        
        
        # Shift X0 to center of mass 
        N = cs.natoms 
        X0 = np.array(X0)
        if X0.shape != (N*3,):
            raise ValueError("X0 must have shape (3*N,) ")
        
        mass = np.array(mass)
        Xcom = np.sum(X0.reshape((N,3)) * mass.reshape((N,1)), axis = 0) / sum(mass)
        
        for i in range(N):
            for j in range(3):
                X0[3*i + j] -= Xcom[j] 
        
        self.cs = cs 
        self.X0 = X0
        self.mass = mass
            
    
    def _csEckartFrame_q2x(self, Q, deriv = 0, out = None, var = None):
        
        # 
        # Eckart frame Q2X function
        #
        # This function is similar to MovingFrameCoordSys
        #
        # The Cartesian coordinates in the original 
        # frame will be evaluated. Then the Eckart
        # rotation matrix and its derivatives will
        # be computed from those using the supplied
        # reference configuration X0.
        #
        # Then the rotation matrix will rotate
        # the original Cartesians
        # 
        
        # Evaluate original coordinate system
        # Use same deriv, out, and var
        #
        X = self.cs.Q2X(Q, deriv = deriv, out = out, var = var)
        # X: (nd, N*3, ...)
        
        base_shape = X.shape[2:]
        nd = X.shape[0] 
        
        if var is None:
            nvar = self.nQ 
        else:
            nvar = len(var) 
        
        N = self.natoms # The number of atoms 
        
        # Subtract the center-of-mass 
        xcom = np.sum(X.reshape((nd,N,3) + base_shape) * self.mass.reshape((1,N,1) + tuple([1]*len(base_shape))), axis = 1) / sum(self.mass)
        # xcom has shape (nd,3,...)
        for i in range(N):
            for j in range(3):
                X[:,3*i + j] -= xcom[:,j] 
        # X is now the center-of-mass frame 
        
        # Make adarray objects for the x,y,z coordinates
        x = [adf.array(X[:,3*i + 0], deriv, nvar) for i in range(N)]
        y = [adf.array(X[:,3*i + 1], deriv, nvar) for i in range(N)]
        z = [adf.array(X[:,3*i + 2], deriv, nvar) for i in range(N)]
        # Now calculate the Eckart rotation matrix
        # using the quaternion method
        #
        # Calculate sum and difference of Cartesians
        xp = [self.X0[3*i + 0] + x[i] for i in range(N)]
        yp = [self.X0[3*i + 1] + y[i] for i in range(N)]
        zp = [self.X0[3*i + 2] + z[i] for i in range(N)]
        
        xm = [self.X0[3*i + 0] - x[i] for i in range(N)]
        ym = [self.X0[3*i + 1] - y[i] for i in range(N)]
        zm = [self.X0[3*i + 2] - z[i] for i in range(N)]
        
        #
        # Evaluate C matrix elements
        # (Eq 24 of reference)
        #
        C11 = sum([self.mass[i] * (xm[i]*xm[i] + ym[i]*ym[i] + zm[i]*zm[i]) for i in range(N)])
        C22 = sum([self.mass[i] * (xm[i]*xm[i] + yp[i]*yp[i] + zp[i]*zp[i]) for i in range(N)])
        C33 = sum([self.mass[i] * (xp[i]*xp[i] + ym[i]*ym[i] + zp[i]*zp[i]) for i in range(N)])
        C44 = sum([self.mass[i] * (xp[i]*xp[i] + yp[i]*yp[i] + zm[i]*zm[i]) for i in range(N)])
        
        C12 = sum([self.mass[i] * (yp[i]*zm[i] - ym[i]*zp[i]) for i in range(N)])
        C13 = sum([self.mass[i] * (xm[i]*zp[i] - xp[i]*zm[i]) for i in range(N)])
        C14 = sum([self.mass[i] * (xp[i]*ym[i] - xm[i]*yp[i]) for i in range(N)])
        C23 = sum([self.mass[i] * (xm[i]*ym[i] - xp[i]*yp[i]) for i in range(N)])
        C24 = sum([self.mass[i] * (xm[i]*zm[i] - xp[i]*zp[i]) for i in range(N)])
        C34 = sum([self.mass[i] * (ym[i]*zm[i] - yp[i]*zp[i]) for i in range(N)])
        
        # Create the derivative array for the C matrix manually 
        dC = np.empty_like(C11.d, shape = (nd,) + base_shape + (4,4)) 
        
        np.copyto(dC[...,0,0], C11.d)
        np.copyto(dC[...,1,1], C22.d)
        np.copyto(dC[...,2,2], C33.d)
        np.copyto(dC[...,3,3], C44.d)
        
        np.copyto(dC[...,0,1], C12.d)
        np.copyto(dC[...,1,0], C12.d)
        
        np.copyto(dC[...,0,2], C13.d)
        np.copyto(dC[...,2,0], C13.d)
        
        np.copyto(dC[...,0,3], C14.d)
        np.copyto(dC[...,3,0], C14.d)
        
        np.copyto(dC[...,1,2], C23.d)
        np.copyto(dC[...,2,1], C23.d)
        
        np.copyto(dC[...,1,3], C24.d)
        np.copyto(dC[...,3,1], C24.d)
        
        np.copyto(dC[...,2,3], C34.d)
        np.copyto(dC[...,3,2], C34.d)
        
        C = adf.array(dC, deriv, nvar)
        
        #
        # Now diagonalize C and get the derivatives
        # of the eigenvector with the lowest eigenvalue
        #
        Lam,V = adf.linalg.eigh_block(C, [0]) 
        
        # Extract quaternion components 
        q0 = adf.array(V.d[...,0,0], deriv, nvar, nck = C.nck, idx = C.idx)
        q1 = adf.array(V.d[...,1,0], deriv, nvar, nck = C.nck, idx = C.idx)
        q2 = adf.array(V.d[...,2,0], deriv, nvar, nck = C.nck, idx = C.idx)
        q3 = adf.array(V.d[...,3,0], deriv, nvar, nck = C.nck, idx = C.idx)
        
        # Calculate rotation matrix 
        
        q00 = q0*q0
        q11 = q1*q1
        q22 = q2*q2
        q33 = q3*q3
        
        q01 = q0*q1
        q02 = q0*q2
        q03 = q0*q3
        q12 = q1*q2
        q13 = q1*q3
        q23 = q2*q3
        
        
        R11 = q00 + q11 - q22 - q33 
        R22 = q00 - q11 + q22 - q33 
        R33 = q00 - q11 - q22 + q33 
        
        R12 = 2*(q12 + q03)
        R13 = 2*(q13 - q02)
        R21 = 2*(q12 - q03)
        R23 = 2*(q23 + q01)
        R31 = 2*(q13 + q02)
        R32 = 2*(q23 - q01)
        
        for i in range(N):
            
            # Calculate new Cartesian coordinates
            # for each atom
            x_new = (R11 * x[i] + R12 * y[i] + R13 * z[i])
            y_new = (R21 * x[i] + R22 * y[i] + R23 * z[i])
            z_new = (R31 * x[i] + R32 * y[i] + R33 * z[i])
            
            # Overwrite the Cartesians for this atom
            np.copyto(X[:,3*i + 0], x_new.d)
            np.copyto(X[:,3*i + 1], y_new.d)
            np.copyto(X[:,3*i + 2], z_new.d)
        
        # Return Eckart frame coordinates
        
        return X 
        
    def diagram(self):
        
        diag = ""
        diag += "     │↓              ↑│        \n"
        diag += "     │            ╔═══╧════╗   \n"
        diag += "     │            ║ Eckart ║   \n"
        diag += "     │            ╚═══╤═╤══╝   \n"
        diag += "     │                │ │↑     \n"
        diag += "     │                │ ╰X0    \n"
        
        Cdiag = self.cs.diagram() # The untransformed CS
        
        return diag + Cdiag 
    
    def __repr__(self):
        
        return f"EckartCoordSys({self.cs!r},{self.X0!r},{self.mass!r})"
    
    
class MovingEckartCoordSys(CoordSys):
    """
    
    A mpving-frame Eckart coordinate system. This method
    uses the quaternion-based algorithm of [Kras2014itdp]_.
    
    
    Attributes
    ----------
    cs : CoordSys
        The original coordinate system
    X0 : CoordSys
        The reference Cartesian configuration as a function of
        the coordinates.
    mass : (N,) ndarray
        The atomic masses used to calculate the 
        Eckart frame
        
        
    
    References
    ----------
    .. [Kras2014itdp] S. Krasnoshchekov, E. Isayeva, and N. Stepanov,
       "Determination of the Eckart molecule-fixed frame by use of the
       apparatus of quaternion algebra," J. Chem. Phys., 140, 154104 (2014).
       https://doi.org/10.1063/1.4870936
     
            
    """
    
    def __init__(self, cs, X0, mass):
        """
        Create a MovingEckartCoordSys object.

        Parameters
        ----------
        cs : CoordSys
            The original coordinate system
        X0 : CoordSys
            The reference Cartesian configuration as a function of
            the coordinates. These will be shifted to the local
            center-of-mass frame.
        mass : (N,) array_like
            The mass of each atom. (This does not have to 
            equal the masses used for KEOs, but it should
            in practice for a useful Eckart frame.)

        """
        
        if not cs.isatomic or not X0.isatomic:
            raise ValueError("Eckart frames must use atomic coordinate systems.")
            
        if cs.nQ != X0.nQ or cs.natoms != X0.natoms:
            raise ValueError("The `cs` and `X0` coordinate systems do not match size.")
            
        # Create new xyz labels
        temp = [["x'{0}".format(i),"y'{0}".format(i),"z'{0}".format(i)] for i in range(cs.natoms) ]
        Xstr = [val for sub in temp for val in sub]
            
        super().__init__(self._csMovingEckartFrame_q2x,
                         nQ = cs.nQ, nX = cs.nX, name = 'Moving Eckart frame',
                         Qstr = cs.Qstr, Xstr = Xstr, isatomic = cs.isatomic,
                         maxderiv = cs.maxderiv, zlevel = None)
        
        self.cs = cs 
        self.X0 = X0
        self.mass = np.array(mass)
        
        
    
    def _csMovingEckartFrame_q2x(self, Q, deriv = 0, out = None, var = None):
        
        # 
        # Moving Eckart frame Q2X function
        #
        # This function is similar to EckartCoordSys
        #
        # The main difference is that the reference configuration
        # Cartesian coordinates (X0) also have non-zero derivatives
        #
        # Moreover, they need to moved to the instantaneous center-of-mass
        # frame.
        #
        
        # Evaluate original coordinate system
        # and the reference configuration
        # Use same deriv, out, and var
        #
        X = self.cs.Q2X(Q, deriv = deriv, out = out, var = var)
        dX0 = self.X0.Q2X(Q, deriv = deriv, var = var)
        
        # X: (nd, N*3, ...)
        # dX0 : " " 
        
        
        base_shape = X.shape[2:]
        nd = X.shape[0] 
        
        if var is None:
            nvar = self.nQ 
        else:
            nvar = len(var) 
        
        N = self.natoms # The number of atoms 
        
        # Subtract the center-of-mass 
        # from both the original coordinate system (X)
        # and the reference configuration  (dX0)
        
        xcom = np.sum(X.reshape((nd,N,3) + base_shape) * self.mass.reshape((1,N,1) + tuple([1]*len(base_shape))), axis = 1) / sum(self.mass)
        # xcom has shape (nd,3,...)
        for i in range(N):
            for j in range(3):
                X[:,3*i + j] -= xcom[:,j] 
                
        xcom = np.sum(dX0.reshape((nd,N,3) + base_shape) * self.mass.reshape((1,N,1) + tuple([1]*len(base_shape))), axis = 1) / sum(self.mass)
        # xcom has shape (nd,3,...)
        for i in range(N):
            for j in range(3):
                dX0[:,3*i + j] -= xcom[:,j]         
                
        # X and dX0 are now in their respective center-of-mass frames
        
        # Make adarray objects for the x,y,z coordinates
        x = [adf.array(X[:,3*i + 0], deriv, nvar) for i in range(N)]
        y = [adf.array(X[:,3*i + 1], deriv, nvar) for i in range(N)]
        z = [adf.array(X[:,3*i + 2], deriv, nvar) for i in range(N)]
        
        # And for the reference configuration coordinates 
        x0 = [adf.array(dX0[:,3*i + 0], deriv, nvar) for i in range(N)]
        y0 = [adf.array(dX0[:,3*i + 1], deriv, nvar) for i in range(N)]
        z0 = [adf.array(dX0[:,3*i + 2], deriv, nvar) for i in range(N)]
        
        # Now calculate the Eckart rotation matrix
        # using the quaternion method
        #
        # Calculate sum and difference of Cartesians
        xp = [x0[i] + x[i] for i in range(N)]
        yp = [y0[i] + y[i] for i in range(N)]
        zp = [z0[i] + z[i] for i in range(N)]
        
        xm = [x0[i] - x[i] for i in range(N)]
        ym = [y0[i] - y[i] for i in range(N)]
        zm = [z0[i] - z[i] for i in range(N)]
        
        #
        # Evaluate C matrix elements
        # (Eq 24 of reference)
        #
        C11 = sum([self.mass[i] * (xm[i]*xm[i] + ym[i]*ym[i] + zm[i]*zm[i]) for i in range(N)])
        C22 = sum([self.mass[i] * (xm[i]*xm[i] + yp[i]*yp[i] + zp[i]*zp[i]) for i in range(N)])
        C33 = sum([self.mass[i] * (xp[i]*xp[i] + ym[i]*ym[i] + zp[i]*zp[i]) for i in range(N)])
        C44 = sum([self.mass[i] * (xp[i]*xp[i] + yp[i]*yp[i] + zm[i]*zm[i]) for i in range(N)])
        
        C12 = sum([self.mass[i] * (yp[i]*zm[i] - ym[i]*zp[i]) for i in range(N)])
        C13 = sum([self.mass[i] * (xm[i]*zp[i] - xp[i]*zm[i]) for i in range(N)])
        C14 = sum([self.mass[i] * (xp[i]*ym[i] - xm[i]*yp[i]) for i in range(N)])
        C23 = sum([self.mass[i] * (xm[i]*ym[i] - xp[i]*yp[i]) for i in range(N)])
        C24 = sum([self.mass[i] * (xm[i]*zm[i] - xp[i]*zp[i]) for i in range(N)])
        C34 = sum([self.mass[i] * (ym[i]*zm[i] - yp[i]*zp[i]) for i in range(N)])
        
        # Create the derivative array for the C matrix manually 
        dC = np.empty_like(C11.d, shape = (nd,) + base_shape + (4,4)) 
        
        np.copyto(dC[...,0,0], C11.d)
        np.copyto(dC[...,1,1], C22.d)
        np.copyto(dC[...,2,2], C33.d)
        np.copyto(dC[...,3,3], C44.d)
        
        np.copyto(dC[...,0,1], C12.d)
        np.copyto(dC[...,1,0], C12.d)
        
        np.copyto(dC[...,0,2], C13.d)
        np.copyto(dC[...,2,0], C13.d)
        
        np.copyto(dC[...,0,3], C14.d)
        np.copyto(dC[...,3,0], C14.d)
        
        np.copyto(dC[...,1,2], C23.d)
        np.copyto(dC[...,2,1], C23.d)
        
        np.copyto(dC[...,1,3], C24.d)
        np.copyto(dC[...,3,1], C24.d)
        
        np.copyto(dC[...,2,3], C34.d)
        np.copyto(dC[...,3,2], C34.d)
        
        C = adf.array(dC, deriv, nvar)
        
        #
        # Now diagonalize C and get the derivatives
        # of the eigenvector with the lowest eigenvalue
        #
        Lam,V = adf.linalg.eigh_block(C, [0]) 
        
        # Extract quaternion components 
        q0 = adf.array(V.d[...,0,0], deriv, nvar, nck = C.nck, idx = C.idx)
        q1 = adf.array(V.d[...,1,0], deriv, nvar, nck = C.nck, idx = C.idx)
        q2 = adf.array(V.d[...,2,0], deriv, nvar, nck = C.nck, idx = C.idx)
        q3 = adf.array(V.d[...,3,0], deriv, nvar, nck = C.nck, idx = C.idx)
        
        # Calculate rotation matrix 
        
        q00 = q0*q0
        q11 = q1*q1
        q22 = q2*q2
        q33 = q3*q3
        
        q01 = q0*q1
        q02 = q0*q2
        q03 = q0*q3
        q12 = q1*q2
        q13 = q1*q3
        q23 = q2*q3
        
        
        R11 = q00 + q11 - q22 - q33 
        R22 = q00 - q11 + q22 - q33 
        R33 = q00 - q11 - q22 + q33 
        
        R12 = 2*(q12 + q03)
        R13 = 2*(q13 - q02)
        R21 = 2*(q12 - q03)
        R23 = 2*(q23 + q01)
        R31 = 2*(q13 + q02)
        R32 = 2*(q23 - q01)
        
        for i in range(N):
            
            # Calculate new Cartesian coordinates
            # for each atom
            x_new = (R11 * x[i] + R12 * y[i] + R13 * z[i])
            y_new = (R21 * x[i] + R22 * y[i] + R23 * z[i])
            z_new = (R31 * x[i] + R32 * y[i] + R33 * z[i])
            
            # Overwrite the Cartesians for this atom
            np.copyto(X[:,3*i + 0], x_new.d)
            np.copyto(X[:,3*i + 1], y_new.d)
            np.copyto(X[:,3*i + 2], z_new.d)
        
        # Return moving Eckart frame coordinates
        
        return X 
        
    def diagram(self):
        
        diag = ""
        diag += "     │↓              ↑│        \n"
        diag += "     │            ╔═══╧════╗   \n"
        diag += "     │            ║ Moving ║   \n"
        diag += "     │            ║ Eckart ║   \n"
        diag += "     │            ╚╤══╤════╝   \n"
        diag += "     │   →         │↑ │        \n"
        diag += "     ├────── X0(Q) ╯  │        \n"
        
        Cdiag = self.cs.diagram() # The untransformed CS
        
        return diag + Cdiag 
    
    def __repr__(self):
        
        return f"MovingEckartCoordSys({self.cs!r},{self.X0!r},{self.mass!r})"
        
        
        
        
        