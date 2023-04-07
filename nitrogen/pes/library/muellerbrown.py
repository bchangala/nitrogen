# -*- coding: utf-8 -*-
"""
Mueller-Brown model surface

The potential parameters are defined 
in Note 7 of Ref [MD1979_].

[MD1979_] K. Mueller and L. D. Brown, ``Location of 
     Saddle Points and Minimum Energy Paths by a 
     Constrained Simplex Optimization Procedure''.
     Theoret. Chim. Acta (Berl.) 53, 75 (1979).
     https://doi.org/10.1007/BF00547608

"""

import nitrogen.autodiff.forward as adf 
import nitrogen as n2 

def Vfun(X, deriv = 0, out = None, var = None):
    
    x,y = n2.dfun.X2adf(X, deriv, var)
    
    A = [-200., -100., -170., 15.]
    
    a = [-1., -1., -6.5, 0.7]
    b = [0., 0., 11., 0.6]
    c = [-10., -10., -6.5, 0.7]
    
    x0 = [1., 0., -0.5, -1.0]
    y0 = [0., 0.5, 1.5, 1.0]
    
    V = 0
    for i in range(4):
        dx = x - x0[i]
        dy = y - y0[i] 
        
        V = V + A[i] * adf.exp(
            a[i]*dx*dx + b[i]*dx*dy + c[i]*dy*dy
            )
        
    return n2.dfun.adf2array([V], out)


######################################
# 
# Define module-scope PES DFun object
#
PES = n2.dfun.DFun(Vfun, nf = 1, nx = 2)
#
#
######################################