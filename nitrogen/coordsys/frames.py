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
           'calcRASangle','calcRASseries']


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
                 int_points = 100):
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
    axis : {'a','b','c'}
        The rotation axis of the reference geometry. This 
        must be an inertial axis.
    int_points : integer, optional
        The number of integration steps. The default is 100. 

    Returns
    -------
    RPAS : (3,3) ndarray
        The rotation axis to the reference PAS system
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
    coordinates :math:`Q_\\mathrm{ref}` and moved to its inertial
    principal axis system (PAS). The rotation matrix which rotates the 
    coordinates fromthe original frame to the reference PAS frame is 
    returned as :math:`\\mathbf{R}_\\mathrm{PAS}`.
    
    The RAS coincides with the reference PAS at the reference geometry.
    As :math:`Q^*` is displaced from its reference value, the RAS frame 
    is rotated relative to the reference PAS by an angle :math:`\\theta(Q^*)` 
    about the principal axis specified by `axis`. The coordinates in the 
    final RAS are thus
    
    ..  math::
        
        \\vec{x}_\\mathrm{RAS} = \\mathbf{R}(\\theta(Q^*)) \\mathbf{R}_\\mathrm{PAS} \\vec{x},
    
    where :math:`\\vec{x}` refers to the original coordinate system `cs`.
    :math:`\\mathbf{R}_\\mathrm{PAS}` orders the principal axes as :math:`(a,b,c)`. If 
    `axis` is ``'a'``, then 
    
    ..  math::
        
        \\mathbf{R}(\\theta) = \\left( \\begin{array}{ccc} 1 & 0 & 0 \\\\
                                      0 & \\cos\\theta & -\\sin\\theta \\\\ 
                                      0 & \\sin\\theta & \\cos\\theta \\end{array} \\right)
            
    and cyclic permutations for ``'b'`` and ``'c'``. Special care should be taken
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
    if axis == 'a':
        rot_axis = 0
    elif axis == 'b':
        rot_axis = 1 
    elif axis == 'c':
        rot_axis = 2 
    else:
        raise ValueError("Unexpected axis (a, b, or c)")
    
    ##############
    # Evaluate the initial coordinate system
    # at the reference geometry 
    #
    X0 = cs.Q2X(Qref)[0] # (N*3,) Cartesian coordinates
    #
    # Calculate the principal axis system of the 
    # reference configuration
    #
    _,RPAS,XCOM  = nitrogen.angmom.X2PAS(X0, mass)

    N = len(X0)//3  # The atom count
    
    # Create a uniform integration grid
    q_initial = Qref[Qstar_idx]
    q_final = Qstar_final_value 
    q_grid = np.linspace(q_initial, q_final, int_points) # The integration grid
    dq = q_grid[1] - q_grid[0] # The step size 
    q_grid_half = q_grid + dq/2 # The half-step grid 
    
    
    # Calculate the coordinates and derivatives in the 
    # reference PAS frame 
    
    mass = np.array(mass) # The masses 
    
    Qgrid = np.tile(Qref, (int_points,1)).T
    Qgrid[Qstar_idx,:] = q_grid
    Xref = cs.Q2X(Qgrid, deriv = 1, var = [Qstar_idx]) # (nd=2, N*3, int_points)
    Xref = np.reshape(Xref, (2, N, 3, int_points))     # (2, N, 3, int_points)
    Xcom = np.sum(Xref * mass.reshape((1,N,1,1)), axis = 1) / sum(mass)
    Xref = Xref - Xcom.reshape((2,1,3,int_points)) # Subtract COM
    Xref = np.einsum('ij,kljn->klin', RPAS, Xref) # Rotate to reference PAS
    
    # Do again for the half-step grid
    Qgrid_half = Qgrid = np.tile(Qref, (int_points,1)).T
    Qgrid_half[Qstar_idx,:] = q_grid_half
    Xref_half = cs.Q2X(Qgrid_half, deriv = 1, var = [Qstar_idx]) # (nd=2, N*3, int_points)
    Xref_half = np.reshape(Xref_half, (2, N, 3, int_points))     # (2, N, 3, int_points)
    Xcom_half = np.sum(Xref_half * mass.reshape((1,N,1,1)), axis = 1) / sum(mass)
    Xref_half = Xref_half - Xcom_half.reshape((2,1,3,int_points)) # Subtract COM
    Xref_half = np.einsum('ij,kljn->klin', RPAS, Xref_half) # Rotate to reference PAS
    
    
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
    
    
    return RPAS, q_grid, theta_grid 
    
    
def calcRASseries(cs, mass, Qref, Qstar_idx, degree, axis):
    
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
    axis : {'a','b','c'}
        The rotation axis of the reference geometry. This 
        must be an inertial axis.

    Returns
    -------
    RPAS : (3,3) ndarray
        The rotation axis to the reference PAS system
    pow : (degree+1,) ndarray
        The power series approximation of the rotation angle :math:`\\theta`
        (in radians) as a function of :math:`Q^*`. 
        
    
    Notes
    -----
    The RAS is defined in [Picket1972]_. See the Notes to :func:`calcRASangle()`
    for more details.
    
    See Also
    --------
    calcRASangle : Calculate the RAS angle via numerical integration. 
    
    References
    ----------
    .. [Picket1972] H. M. Pickett, "Vibration-Rotation Interactions and the Choice of 
       Rotating Axes for Polyatomic Molecules," J. Chem. Phys., 56, 1715 (1972).
       https://doi.org/10.1063/1.1677430
     
    
    """
    
    ###########################
    # Calculate the RAS angle via a power series solution
    #
    
    #################
    # Parse rotation axis
    #
    if axis == 'a':
        rot_axis = 0
    elif axis == 'b':
        rot_axis = 1 
    elif axis == 'c':
        rot_axis = 2 
    else:
        raise ValueError("Unexpected axis (a, b, or c)")
        
    pow_order = degree # Maximum power in theta(q) power series 
    nd = pow_order + 1 # The number of derivatives (including zeroth)
    N = len(mass)      # The number of atoms, N 
    
    # Calculate the coordinates and their derivatives w.r.t. Q*
    dX = cs.Q2X(Qref, deriv = pow_order, var = [Qstar_idx]) # (nd, N*3)
    _,RPAS,_ = nitrogen.angmom.X2PAS(dX[0], mass) # Calculate PAS frame
    dX = np.reshape(dX, (nd, N, 3)) # (nd,N,3)
    
    
    # Subtract center-of-mass 
    Xcom = np.sum( dX * np.array(mass).reshape(1,N,1), axis = 1) / sum(mass) 
    dX = dX - Xcom.reshape((nd,1,3))
    dX0 = np.einsum('ij,klj->kli',RPAS,dX) # (nd,N,3)
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
    
    return RPAS, theta_pow 
    