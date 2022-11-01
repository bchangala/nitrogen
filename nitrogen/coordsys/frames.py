# -*- coding: utf-8 -*-
"""
frames.py

Body-frame rotation and transformation
"""

from .coordsys import CoordSys, CoordTrans
import numpy as np 
import nitrogen.autodiff.forward as adf 
import nitrogen.dfun 


__all__ = ['PermutedAxisCoordSys', 'RotatedCoordSys', 'MovingFrameCoordSys']


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
    