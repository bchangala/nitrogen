# -*- coding: utf-8 -*-
"""
symmetry.py

Symmetry operations

"""

import numpy as np 
from scipy.sparse.linalg import LinearOperator 


class GenericSymmetryOperator(LinearOperator):
    """
    
    A general-case symmetry operator for
    direct-product-basis vectors.
    
    =============   ==========================
    `op` tuple      Description
    =============   ==========================
    ('swap',i,j)    Swap axes `i` and `j`.
    ('flip',i)      Flip axis `i`.
    ('diag',i,v)    Multiply a diagonal operator with diagonal elements `v`
                    along axis `i`.
    =============   ==========================
                    
    """
    
    def __init__(self,factor_shape, ops):
        """
        
        Parameters
        ----------
        factor_shape : tuple
            The direct product basis shape, (`n1`, `n2`, ...).
        ops : list of tuple
            The operations to perform. See description in class notes.
            
        """
        
        N = 1 
        for d in factor_shape:
            N = N * d  # Total size 
        
        self.shape = (N,N)
        self.dtype = np.float64 
        self.factor_shape = factor_shape 
        
        self.N = N 
        self.ops = ops 
        
        
    def _matvec(self, x):
        """
        The matrix-vector product 
        """
        
        # Reshape y to the direct-product basis shape.
        y = x.copy().reshape(self.factor_shape)
        
        # Now perform all operations in `ops` list.
        for op in self.ops :
            
            if op[0] == 'swap':
                #
                # Swap axis i and j
                #
                y = np.swapaxes(y, op[1], op[2])
            elif op[0] == 'flip':
                #
                # Flip axis i
                #
                y = np.flip(y, op[1])
            elif op[0] == 'diag':
                #
                # Multiply axis i with diagonal elements.
                #
                # We will use ndarray broadcasting to do this
                #
                bshape = [1] * len(self.factor_shape)
                bshape[op[1]] = self.factor_shape[op[1]] 
                diag_op = op[2].reshape(tuple(bshape))
                y = diag_op * y 
        
        # Reshape y to 1-d 
        #
        y = np.reshape(y, (self.N,))
        
        return y 
    
class SymmetryProjector(LinearOperator):
    """
    A symmetry projection operator,
    
    ..  math::
        
        P = \\frac{1}{h} \\sum_i \\chi_i R_i
    
    """
    
    def __init__(self,R, chars):
        """
        
        Parameters
        ----------
        R : list of LinearOperator
            The symmety operations :math:`R_i`, **excluding** identity.
        chars : array_like
            The characters :math:`\\chi_i` of the representation, **including** identity, 
            which must be the first element of `chars`.
            
        """
        
        N = R[0].shape[0]
        self.shape = (N,N)
        self.N = N 
        self.dtype = R[0].dtype 
        self.R = R 
        self.order = len(chars) # The total number of operators
        self.chars = np.array(chars)
        
        
    def _matvec(self, x):
        
        chars = self.chars
        
        # Start with identity
        y = chars[0] * x.copy().reshape((self.N,))
        
        # Proceed
        for i in range(1,self.order):
            y += chars[i] * self.R[i-1].matvec(x) 
            
        y /= self.order 
        
        return y 
       
