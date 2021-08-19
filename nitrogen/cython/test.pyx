# -*- coding: utf-8 -*-
"""
Created on Mon Aug  2 19:11:18 2021

@author: bryan
"""

cimport cython 

def test(double [:] X):
    
    cdef size_t N = X.shape[0]
    cdef size_t i
    cdef double sum = 0.0 
    
    for i in range(N):
        sum += X[i]
    
    return sum


@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
@cython.nonecheck(False)    # Deactive None checking
def naive_matmul(double [:,:] X, double [:,:] Y, double [:,:] Z):
    
    cdef:
        size_t n = X.shape[0]
        size_t p = X.shape[1]
        size_t m = Y.shape[1] 
        
        size_t i,j,k 
        
        
    for i in range(n):
        for j in range(m):
            
            Z[i,j] = 0.0
            for k in range(p):
                Z[i,j] += X[i,k] * Y[k,j] 
    
    return
                
        