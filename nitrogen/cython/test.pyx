# -*- coding: utf-8 -*-
"""
Created on Mon Aug  2 19:11:18 2021

@author: bryan
"""

def test(double [:] X):
    
    cdef size_t N = X.shape[0]
    cdef size_t i
    cdef double sum = 0.0 
    
    for i in range(N):
        sum += X[i]
    
    return sum