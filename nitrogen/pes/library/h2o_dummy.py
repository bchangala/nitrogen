"""
h2o_dummy.py

A water-like surface 

       
"""

import nitrogen as n2
import numpy as np
import nitrogen.autodiff.forward as adf


def Vfun(X, deriv = 0, out = None, var = None):
    """
    expected atom ordering H O H
    """
    x = n2.dfun.X2adf(X, deriv, var)
    
    H1 = [x[0], x[1], x[2]]
    O  = [x[3], x[4], x[5]]
    H2 = [x[6], x[7], x[8]]
    
    R1 = [ H1[i]-O[i] for i in range(3)]
    R2 = [ H2[i]-O[i] for i in range(3)]
    
    r1 = adf.sqrt(R1[0]*R1[0] + R1[1]*R1[1] + R1[2]*R1[2])
    r2 = adf.sqrt(R2[0]*R2[0] + R2[1]*R2[1] + R2[2]*R2[2])
    cosa  = R1[0]*R2[0] + R1[1]*R2[1] + R1[2]*R2[2] / (r1*r2)
    
    re = 1.00     # Angstrom
    ae = 105.00 * n2.pi/180.0 # radians
    ce = np.cos(ae) 
    
    # Stretching
    V = Vmorse(r1, 3600.00, 1.0, 50000.0, re)
    V += Vmorse(r2, 3600.00, 1.0, 50000.0, re)
    # Bend
    dce = cosa - ce
    V += 1.6e4 * (dce*dce)
    
    
    return n2.dfun.adf2array([V], out)

def Vmorse(r,hbomega,mu,De,re):
    """
    """
    k = mu * (hbomega/n2.constants.hbar)**2 # k, force constant
    a = np.sqrt(k / (2*De)) # Morse 'a' constant 
    
    temp = (1.0 - adf.exp(-a * (r-re)))
    Vm = De * (temp*temp)
    
    return Vm
    
#################################
#
# Create module-scope DFun object
PES = n2.dfun.DFun(Vfun, nf = 1, nx = 9)
#
#################################



