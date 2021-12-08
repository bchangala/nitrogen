"""
h3cation_mbb1986.py

H3^+ ground state surface from Ref [1]_ ("MBB"). Note the issues
mentioned in Ref [2]_. Comprehensive energies for this surface
can be found in Ref [3]_.

Several fits and versions are reported in Ref [1]_. This module implements
the 87CGTO, N=7 surface (Table IV), with Re = 1.6504 bohr and beta = 1.30.
It includes scaling of the Se**2 term by 1.0019.


References
----------

.. [1] W. Meyer, P. Botschwina, and P. Burton, J. Chem. Phys. 84, 891 (1986).
       https://doi.org/10.1063/1.450534
       
.. [2] M. J. Bramley, J. R. Henderson, J. Tennyson, and B. T. Sutcliffe,
       J. Chem. Phys. 98, 10104 (1993). 
       https://doi.org/10.1063/1.464402
       
.. [3] M. J. Bramley, J. W. Tromp, T. Carrington Jr., G. C. Corey,
       J. Chem. Phys. 100, 6175 (1994).
       https://doi.org/10.1063/1.467273
       
"""

import nitrogen as n2
import numpy as np
import nitrogen.autodiff.forward as adf


def Vfun(X, deriv = 0, out = None, var = None):
    """
    
    """
    x = n2.dfun.X2adf(X, deriv, var)
    
    H1 = [x[0], x[1], x[2]]
    H2 = [x[3], x[4], x[5]]
    H3 = [x[6], x[7], x[8]]
    
    R1 = [ H2[i]-H3[i] for i in range(3)]
    R2 = [ H3[i]-H1[i] for i in range(3)]
    R3 = [ H1[i]-H2[i] for i in range(3)]
    
    r1 = adf.sqrt(R1[0]*R1[0] + R1[1]*R1[1] + R1[2]*R1[2])
    r2 = adf.sqrt(R2[0]*R2[0] + R2[1]*R2[1] + R2[2]*R2[2])
    r3 = adf.sqrt(R3[0]*R3[0] + R3[1]*R3[1] + R3[2]*R3[2])
    
    re = 1.6504 * n2.constants.a0 # Angstroms 
    beta = 1.30 # Morse parameter 
    
    rp1 = morse_scale(r1, re, beta)
    rp2 = morse_scale(r2, re, beta)
    rp3 = morse_scale(r3, re, beta)

    Sa = (rp1 + rp2 + rp3) / np.sqrt(3.0) 
    Sx = (2*rp3 - rp1 - rp2) / np.sqrt(6.0)
    Sy = (rp1 - rp2) / np.sqrt(2.0)
    
    Sx2 = Sx*Sx 
    Sy2 = Sy*Sy
    
    Se2 = Sx2 + Sy2 # Se^2
    Se4 = Se2*Se2 
    Se6 = Se4*Se2
    
    Sb3 = Sx * ( Sx2 - 3 * Sy2)
    Sc3 = Sy * ( Sy2 - 3 * Sx2)
    
    ONE = 0*Sa + 1.0 # one
    Sa2 = Sa * Sa 
    Sa3 = Sa2 * Sa
    Sa4 = Sa2 * Sa2 
    Sa5 = Sa4 * Sa 
    Sa6 = Sa3 * Sa3 
    Sa7 = Sa6 * Sa 
    
    t_n = [ONE, Sa, Sa2, Sa3, Sa4, Sa5, Sa6, Sa7] # n = 0, 1, 2, ..., 7
    t_2m = [ONE, Se2, Se4, Se6 ]        # m = 0, 1, 2, 3
    t_3k = [ONE, Sb3, Sb3*Sb3 - Sc3*Sc3] # k = 0, 1, 2
    
    # Compute surface expansion
    v = expansion(t_n, t_2m, t_3k)
    
    return n2.dfun.adf2array([v], out)

######################################
# 
# Define module-scope PES DFun object
#
PES = n2.dfun.DFun(Vfun, nf = 1, nx = 9)
#
#
######################################


def morse_scale(r, re, beta):
    return (1.0-adf.exp(-beta * (r/re - 1.0)))/beta

def expansion(t_n, t_2m, t_3k):
    
    # 87 CGTO N = 7 
    # Table IV of Ref [1]
    #
    # Note that I use (n, m, k)
    # directly, instead of 
    # (n, 2m, 3k)
    #
    # This expansion is in units of
    # 10**-6 a.u.
    # 
    surf = [
        (0, 0, 0,      0.),
        (1, 0, 0,    130.),
        (2, 0, 0, 204603.),
        (0, 1, 0, 266219. * 1.0019), # Se^2 term, scaled
        (3, 0, 0, -49832.),
        (1, 1, 0,-241851.),
        (0, 0, 1,  -6490.),
        (4, 0, 0,  25002.),
        (2, 1, 0, 131115.),
        (1, 0, 1,  88684.),
        (0, 2, 0,  44851.),
        (5, 0, 0,  -2115.),
        (3, 1, 0, -50919.),
        (2, 0, 1, -28688.),
        (1, 2, 0, -11820.),
        (0, 1, 1,  -3185.),
        (6, 0, 0,   4346.),
        (4, 1, 0,  50424.),
        (3, 0, 1,  57028.),
        (2, 2, 0, 120688.),
        (1, 1, 1,  73273.),
        (0, 3, 0,  15068.),
        (0, 0, 2,   -339.),
        (7, 0, 0,   -277.),
        (5, 1, 0,    887.),
        (4, 0, 1,   9333.),
        (3, 2, 0,  23840.),
        (2, 1, 1, 104361.),
        (1, 3, 0,  37493.),
        (1, 0, 2,  -3238.),
        (0, 2, 1,   7605.)
        ]
    
    V = 0.0 
    
    for term in surf:
        V = V + term[3] * (t_n[ term[0] ] * t_2m[ term[1]] * t_3k[term[2]] )
    
    V = V * (n2.constants.Eh * 1e-6)
    return V