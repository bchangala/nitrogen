"""
H2O_PJT1996.py

Water potential energy surface from Ref [1]_ ("PJT2"). This
surface was optimized for H2(^16O), containing contributions from B-O breakdown.
Some coordinate definitions are required from Ref [2]_.


References
----------
.. [1] O. L. Polyanksy, P. Jensen, and J. Tennyson, J. Chem. Phys. 105, 6490 (1996).
       https://doi.org/10.1063/1.472501 
.. [2] P. Jensen, J. Mol. Spectrosc. 128, 478 (1988).
       https://doi.org/10.1016/0022-2852(88)90164-6
       
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
    
    re = 0.9579205     # Angstrom
    ae = 104.4996470 * n2.pi/180.0 # radians
    aM = 2.226000      # Angstrom**-1
    
    dq0 = 1.0 - adf.exp(-aM * (r1 - re))
    dq1 = 1.0 - adf.exp(-aM * (r2 - re))
    dq2 = -np.cos(ae) + cosa
    
    dq = [dq0, dq1, dq2]
    
    v = V0(dq)
    v = v + F1(dq)
    v = v + F234(dq) 
    v = v + Fn(dq)
    
    return n2.dfun.adf2array([v], out)


PES = n2.dfun.DFun(Vfun, nf = 1, nx = 9)

def V0(dq):
    """ dq : list of adarray """
    
    f0 = [0.,
          0.,
	   18902.4419343, 
		1893.9978814, 
		4096.7344377, 
	   -1959.6011328, 
	    4484.1589338, 
		4044.5538881, 
	   -4771.4504354]

    temp = dq[2] * dq[2]
    
    val = 0.0 * dq[2] # zero
    
    for i in range(2,len(f0)):
        val = val +  f0[i] * temp 
        temp = temp * dq[2]
    
    return val

def F1(dq):
    """ dq : list of adarray """
    
    f1 = [ 0,
        -6152.4014118,
        -2902.1391226,
        -5732.6846068,
          953.8876083 
        ]
    
    temp = 1.0 * dq[2]
    
    val = 0.0 * dq[2] # zero
    
    temp2 = dq[0] + dq[1]
    for i in range(1,len(f1)):
        val = val + (f1[i] * temp) * temp2
        temp = temp * dq[2]
    
    return val

def F234(dq):
    """ dq : list of adarray """
    
    f11 = [42909.8886909,
           -2767.1919717,
           -3394.2470551]
    
    f13 = [-1031.9305520,
           6023.8343525,
           0.0]
    
    f111 = [0.0,
            124.2352938,
            -1282.5066122]
    
    f113 = [-1146.4910952,	
            9884.4168514,	
            3040.3402183]
    
    f1111 = [2040.9674526,
             0.0,
             0.0]
    
    f1113 = [-422.0339419,
             -7238.0997940,
             0.0]

    val = 0.0 * dq[2] # zero
    
    temp = 1.0 + val  # one
    
    t11 = dq[0] * dq[0] + dq[1] * dq[1]
    t13 = dq[0] * dq[1]
    t111 = dq[0]*dq[0]*dq[0] + dq[1]*dq[1]*dq[1]
    t113 = dq[0]*dq[0]*dq[1] + dq[0]*dq[1]*dq[1]
    t1111 = dq[0]*dq[0]*dq[0]*dq[0] + dq[1]*dq[1]*dq[1]*dq[1]
    t1113 = dq[0]*dq[0]*dq[0]*dq[1] + dq[0]*dq[1]*dq[1]*dq[1] 
    
    
    for i in range(len(f11)):
        val = val + f11[i] * temp * t11
        val = val + f13[i] * temp * t13
        val = val + f111[i] * temp * t111
        val = val + f113[i] * temp * t113
        val = val + f1111[i] * temp * t1111
        val = val + f1113[i] * temp * t1113
        
        temp = temp * dq[2]

    return val

def Fn(dq):
    """ dq : list of adarray """
    
    f11111 = -4969.2454493
    f111111 = 8108.4965235
    f1111111 = 90.0000000

    t5a = dq[0]*dq[0]*dq[0]*dq[0]*dq[0]
    t5b = dq[1]*dq[1]*dq[1]*dq[1]*dq[1]
    
    t6a = t5a * dq[0]
    t6b = t5b * dq[1]
    
    t7a = t6a * dq[0]
    t7b = t6b * dq[1]
    
    val = f11111*(t5a + t5b)
    val = val + f111111 * (t6a + t6b)
    val = val + f1111111 * (t7a + t7b)
    
    return val

