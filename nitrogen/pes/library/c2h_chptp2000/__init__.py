"""
c2h_chptp2000

Multistate diabatic Hamiltonian for the :math:`X{}^2\\Sigma^+` and 
:math:`A{}^2\\Pi` states of C\ :sub:`2`\ H from Ref [1]_. 
The `PES` DFun object evaluates the
diabatic V matrix, returning the lower triangle in row major order.
The state ordering is :math:`\\Sigma (A'), \\Pi (A'), \\Pi (A'')`\ .
The input coordinates are :math:`R_\\text{CC}, R_\\text{CH}, \\theta`, in 
Angstrom and degree units. The output units are :math:`\\text{cm}^{-1}`.

References
----------

.. [1] S. Carter, N. C. Handy, C. Puzzarini, R. Tarroni, and P. Palmieri,
       Mol. Phys. 98, 1697 (2000).
       https://doi.org/10.1080/00268970009483375

       
"""

import nitrogen as n2
import numpy as np
import nitrogen.autodiff.forward as adf

import os

#
# Import Table 1 of Ref [1]
#
Cijk = np.loadtxt(os.path.join(os.path.dirname(__file__), 'table1.txt'))
                  
RCC_ref = [2.284647, 2.432802, 2.432802, 2.500000]
RCH_ref = [2.007159, 2.019287, 2.019287, 2.000000]
theta_ref = [180.0, 180.0, 180.0, 180.0]

def Vfun(X, deriv = 0, out = None, var = None):
    """
    expected order RCC (Angstroms), RCH (Angstroms), theta (degrees)
    """
    x = n2.dfun.X2adf(X, deriv, var)
    
    a0 = n2.constants.a0
    
    RCC = x[0] / a0 # C--C bond length, convert angstroms to bohr
    RCH = x[1] / a0 # C--H bond length, convert angstroms to bohr
    theta = x[2]    # C--C--H angle, degrees
    
    n = Cijk.shape[0] # The number of expansion terms 
    
    # For each surface function
    v = []
    for p in range(4):
        
        dCC = RCC - RCC_ref[p]
        dCH = RCH - RCH_ref[p]
        dtheta = (theta - theta_ref[p]) * (n2.pi/180.0) # convert to radians
        
        dCC_pow = pows(dCC, round(max(Cijk[:,0])))
        dCH_pow = pows(dCH, round(max(Cijk[:,1])))
        dtheta_pow = pows(dtheta, round(max(Cijk[:,2])))
        
        vp = 0
        for i in range(n):
            
            c = Cijk[i,p + 3] # The coefficient of this surface
            
            vp += c * dCC_pow[round(Cijk[i,0])] * \
                      dCH_pow[round(Cijk[i,1])] * \
                      dtheta_pow[round(Cijk[i,2])]
                
        
        
        if p == 3: # The V12 surface
            vp *= adf.sin( dtheta ) * \
                  (1.0 - adf.tanh(dCC)) * (1.0 - adf.tanh(dCH))
        
        # Convert from Hartree to cm^-1
        vp *= n2.constants.Eh 
        
        v.append(vp)
    
    # v contains [V11, V22, V33, V12]
    # Now build the lower triangle
    # in row major order: 
    # V11
    # V21 = V12,  V22
    # V31 = 0, V32 = 0, V33
    #
    V31 = adf.const_like(0.0,v[0])
    V32 = adf.const_like(0.0,v[0])
    
    lower = [v[0], v[3], v[1], V31, V32, v[2]]
  
    return n2.dfun.adf2array(lower, out)

def pows(x, pmax):
    
    if pmax < 0:
        raise ValueError("pmax must be non-negative integer")
    xp = [None for i in range(pmax+1)]
    xp[0] = 1.0
    for i in range(1, pmax+1):
        xp[i] = x * xp[i-1] 
    return xp 

######################################
# 
# Define module-scope PES DFun object
#
PES = n2.dfun.DFun(Vfun, nf = 6, nx = 3)
#
#
######################################