#cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True, language_level=3
"""
Simple Cython implementation of forward AD

cyad_core.pyx

Implementation source file

"""
from libc.stdlib cimport malloc, free 
from libc.math cimport sqrt as csqrt
from libc.math cimport exp as cexp  
from libc.stdio cimport printf

from .cyad_core cimport adtab 


cdef void mul(double *Z, double *X, double *Y, adtab *t):
    """
    Simple product, Z = X * Y
    
    """
    cdef size_t i
        
    # Initialize result 
    for i in range(t.nd):
        Z[i] = 0.0 
    
    # Compute product
    for i in range(t.table_size):
        Z[t.idxZ[i]] += X[t.idxX[i]] * Y[t.idxY[i]]
    
    return 

cdef void mulacc(double *Z, double *X, double *Y, adtab *t):
    """
    Simple product with accumulation
    
    Z += X * Y 
    """
    
    cdef size_t i
            
    # Compute product
    for i in range(t.table_size):
        Z[t.idxZ[i]] += X[t.idxX[i]] * Y[t.idxY[i]]
    
    return

cdef void add(double *Z, double *X, double *Y, adtab *t):
    """
    Add, Z = X + Y
    """
    cdef int i
    
    for i in range(t.nd):
        Z[i] = X[i] + Y[i] 
    
    return

cdef void smul(double *Z, double s, double *X, adtab *t):
    """
    Scalar multiplication
    
    Z = s * X 
    """
    
    cdef size_t i 
    
    for i in range(t.nd):
        Z[i] = s * X[i]
        
    return

cdef void smulacc(double *Z, double s, double *X, adtab *t):
    """
    Scalar multiplication with accumulation 
    
    Z += s * X
    """
    cdef size_t i 
    for i in range(t.nd):
        Z[i] += s * X[i]
    
    return

cdef void sub(double *Z, double *X, double *Y, adtab *t):
    """
    Simple subtraction, Z = X - Y 
    """
    
    cdef size_t i 
    
    for i in range(t.nd):
        Z[i] = X[i] - Y[i] 
    
    return  

cdef void sqrt(double *Z, double *X, double *F, double **temp, adtab *t):
    """
    Square root, Z = sqrt[X]
    
    Workspace
    ---------
    F : (k+1,) double 
    temp : (3,nd) double 
    
    """
    
    #
    # Note: this implementation could be made
    # more efficient using recursive
    # expression for sqrt derivatives in the 
    # same way as for matrix sqrt.
    #
     
    # compute the scaled derivatives of sqrt[] 
    # evaluated at X0 
    #
    # cdef double *F = malloc1d(t.k + 1) 
    # cdef double **temp = malloc2d(3,t.nd)
    # free2d(temp,3)
    # free1d(F) 
    
    cdef int i 
    
    # Calculate scaled derivatives of sqrt[] 
    F[0] = csqrt(X[0]) # Value 
    
    # Derivatives
    for i in range(1, t.k + 1):
        F[i] = F[i-1] * (1.5 - i) / (X[0] * i) 
    
    chain1d(Z, X, F, temp, t)

    
    return 

cdef void exp(double *Z, double *X, double *F, double **temp, adtab *t):
    """
    Exponential function, Z = exp[X]
    
    Workspace
    ---------
    F : (k+1,) double 
    temp : (3,nd) double 
    
    """
    # compute the scaled derivatives of exp
    # evaluated at X0 
    #
    cdef int i 
    
    # Calculate scaled derivatives of exp[]
    F[0] = cexp(X[0]) # Value 
    for i in range(1, t.k + 1):
        F[i] = F[i-1] / i 
    
    chain1d(Z, X, F, temp, t)
    
    return 

cdef void chain1d(double *Z, double *X, double *F, double **temp, adtab *t):
    """
    Computes derivatives of analytic univariate 
    function via chain rule
    
    Z = F(X) 
    
    F ... scaled derivatives of F w.r.t its argument evaluated at X 
    
    Workspace
    ---------
    temp : (3,nd) double
    
    """
    
    cdef double X0 
    cdef size_t i,k
    
    # Initialize Z 
    Z[0] = F[0] # Value 
    for i in range(1,t.nd):
        Z[i] = 0.0 
        
    
    for i in range(t.nd):
        temp[0][i] = X[i] # copy X 
        temp[1][i] = X[i] 
    temp[0][0] = 0.0 # Remove value
    temp[1][0] = 0.0 
    
    # temp[0] and temp[1] stores the displacement expression 
    
    if t.k > 0:
        # First degree of expansion
        # Z <-- F[1] * (X - X0)
        smulacc(Z, F[1], temp[1], t) 
        
        # Continue with higher orders of expansion 
        for i in range(2, t.k + 1):
            # Compute next power of displacement 
            mul(temp[2], temp[1], temp[0], t)
            
            # Add contribution to Z 
            smulacc(Z, F[i], temp[2], t) 
            
            for k in range(t.nd):
                temp[1][k] = temp[2][k] # move for next iteration 
                
    
    # Z is complete thru order t.k 
    
    return 
    

#
# Memory allocation and freeing
# utility functions
#
cdef double *malloc1d(size_t n):
    return <double *>malloc(n * sizeof(double))

cdef double **malloc2d(size_t m, size_t n):
    
    cdef:
        size_t i 
        double **A
        
    A = <double **> malloc(m * sizeof(double *))
    for i in range(m):
        A[i] = <double *>malloc(n * sizeof(double))
    
    return A

cdef void free1d(double *ptr):
    free(ptr)
    return 

cdef void free2d(double **ptr, size_t m):
    cdef size_t i 
    for i in range(m):
        free(ptr[i])
    free(ptr)
    return 

