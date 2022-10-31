# -*- coding: utf-8 -*-
"""
frames.py

Body-frame rotation and transformation
"""

from .coordsys import CoordSys, CoordTrans
import numpy as np 


__all__ = ['PermutedAxisCoordSys']


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
                