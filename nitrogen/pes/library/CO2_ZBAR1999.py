"""
CO2_ZBAR1999.py

CO_2 ground state surface from Ref [1]_, using the PES 
form presented in Ref [2]_. This PES is refined with
respect to a large number of experimental vibrational frequencies.

Input: Cartesians in C--O--C order (Angstroms)
Output: energy in cm^-1

References
----------

.. [1] J. Zúñiga, A. Bastida, M. Alacid, A. Requena, Chem. Phys. Lett. 313,
       670 (1999). 
       https://doi.org/10.1016/S0009-2614(99)01080-5
       
.. [2] J. N. Murrell and H. Guo, J. Chem. Soc. Faraday Trans. 2, 83, 683 (1987).
       https://doi.org/10.1039/F29878300683

"""



import nitrogen as n2
import nitrogen.autodiff.forward as adf


def Vfun(X, deriv = 0, out = None, var = None):
    """
    expected order C--O--C
    """
    x = n2.dfun.X2adf(X, deriv, var)
    
    O1 = [x[0], x[1], x[2]]
    C =  [x[3], x[4], x[5]]
    O2 = [x[6], x[7], x[8]]
    
    R1 = [ O1[i]-C[i] for i in range(3)]
    R2 = [ O2[i]-C[i] for i in range(3)]
    R3 = [ O1[i]-O2[i] for i in range(3)]
    
    r1 = adf.sqrt(R1[0]*R1[0] + R1[1]*R1[1] + R1[2]*R1[2])
    r2 = adf.sqrt(R2[0]*R2[0] + R2[1]*R2[1] + R2[2]*R2[2])
    r3 = adf.sqrt(R3[0]*R3[0] + R3[1]*R3[1] + R3[2]*R3[2])
    
    VO1 = 1.967 * n2.constants.eV 
    
    v = VO1 * (f1(r1,r2,r3) + f2(r1,r2,r3)) + \
        V2CO(r1) + V2CO(r2) + V2OO(r3) + \
        V3OCO(r1,r2,r3)
    
    return n2.dfun.adf2array([v], out)

######################################
# 
# Define module-scope PES DFun object
#
PES = n2.dfun.DFun(Vfun, nf = 1, nx = 9)
#
#
######################################

def f1(R1,R2,R3):
    """ f1 switching function """
    
    R01,R02,R03 = 1.160, 1.160, 2.320 
    alpha1 = 1.0
    
    rho1 = R1 - R01 
    rho2 = R2 - R02 
    rho3 = R3 - R03 
    
    return 0.5 * (1 - adf.tanh(0.5 * alpha1 * (3*rho1 - rho2 - rho3)))
 
def f2(R1,R2,R3):
    """ f2 switching function """
    
    R01,R02,R03 = 1.160, 1.160, 2.320 
    alpha2 = 1.0
    
    rho1 = R1 - R01 
    rho2 = R2 - R02 
    rho3 = R3 - R03 
    
    return 0.5 * (1 - adf.tanh(0.5 * alpha2 * (3*rho2 - rho1 - rho3)))   

def V2CO(R):
    """ CO diatomic energy"""
    
    De = 11.226 * n2.constants.eV 
    re = 1.128 # Angstroms 
    a1,a2,a3 = 3.897, 2.305, 1.898 
    
    r = R - re
    r2 = r * r 
    r3 = r2 * r
    
    V2 = -De * (1 + a1*r + a2*r2 + a3*r3) * adf.exp(-a1*r) 
    
    return V2 

def V2OO(R):
    """ O2 diatomic energy"""
    
    De = 5.213 * n2.constants.eV 
    re = 1.208 # Angstroms 
    a1,a2,a3 = 6.080, 11.477, 11.003
    
    r = R - re
    r2 = r * r 
    r3 = r2 * r
    
    V2 = -De * (1 + a1*r + a2*r2 + a3*r3) * adf.exp(-a1*r) 
    
    return V2 

def V3OCO(R1,R2,R3):
    """ O--C--O three body term """
     
    R = [R1, R2, R3]
    gamma = [2.357, 2.357, 0.959]
    R0 = [1.160, 1.160, 2.320]
    
    rho = [R[i] - R0[i] for i in range(3)]
    
    fact = [1.0 - adf.tanh(0.5 * gamma[i] * rho[i]) for i in range(3)]
    
    return P3(rho) * fact[0] * fact[1] * fact[2] 

def P3(rho):
    
    V0 = 3.7398 * n2.constants.eV 
    
    rho1 = [1, rho[0], rho[0] * rho[0], rho[0] * rho[0] * rho[0]]
    rho2 = [1, rho[1], rho[1] * rho[1], rho[1] * rho[1] * rho[1]]
    rho3 = [1, rho[2], rho[2] * rho[2], rho[2] * rho[2] * rho[2]]
    
    C1 = C2 = 2.783500
    C3 = -2.278700
    C11 = C22 = 1.087581
    C33 = 0.806475
    C12 = 1.260657 
    C13 = C23 = -0.331311 
    C111 = C222 = -0.129443 
    C333 = -0.858272 
    C112 = C122 = 1.522264 
    C113 = C223 = 0.673072 
    C133 = C233 = 0.554937
    C123 = 0.765558 
    
    P = V0 * (1 + C1 * rho1[1] + C2 * rho2[1] + C3 * rho3[1] + \
              C11 * rho1[2] + C22 * rho2[2] + C33 * rho3[2] + \
              C12 * rho1[1] * rho2[1] + C13 * rho1[1] * rho3[1] + C23 * rho2[1] * rho3[1] + \
              C111 * rho1[3] + C222 * rho2[3] + C333 * rho3[3] + \
              C112 * rho1[2] * rho2[1] + C122 * rho1[1] * rho2[2] + \
              C113 * rho1[2] * rho3[1] + C223 * rho2[2] * rho3[1] + \
              C133 * rho1[1] * rho3[2] + C233 * rho2[1] * rho3[2] + \
              C123 * rho1[1] * rho2[1] * rho3[1])
    
    return P
    