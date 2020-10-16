"""
nitrogen.linalg.packed
-------------------------

Packed-storage matrices and routines.
The lower triangle is stored in row-major order.
(For symmetric matrices, this is equivalent to upper triangle
 column-major order. For Hermitian matrices, this is
 the conjugate of the upper triangle in column-major order.)

"""

import numpy as np

def k2IJ(k):
    """
    Calculate the full 2D index (`I`,`J`) for a given
    packed 1D index `k`. The returned index is always
    in the lower triangle.

    Parameters
    ----------
    k : int
        The packed 1D index.

    Returns
    -------
    I,J : np.uint64
        The unpacked 2D index.

    """
    
    I = np.uint64((np.sqrt(8*k+1)-1)/2)
    J = np.uint64(k - (I*(I+1))//2)
    
    return I,J

def IJ2k(I,J):
    """
    Assuming a symmetric array, calculate the 1D
    packed storage index for the (I,J) = (J,I) element
    """
    
    # Let (i,j) be the equivalent position
    # in the lower triangle
    i = max(I,J)
    j = min(I,J)
    
    k = np.uint64( (i*(i+1))//2 + j)
    
    return k

def n2N(n):
    """
    Calculate the square matrix rank N
    for a packed storage size n

    Parameters
    ----------
    n : int
        The packed length.

    Returns
    -------
    N : np.uint64
        The matrix rank.

    """
    
    N = np.uint64((np.sqrt(8*n+1)-1)/2)
    
    return N

