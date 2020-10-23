"""
nitrogen.angmom
---------------

Angular momentum and spherical tensor routines.

"""

import numpy as np

def Jbf_cs(J):
    """
    Calculate Condon-Shortley body-fixed J operators

    Parameters
    ----------
    J : int
        Angular momentum: 0, 1, 2, ...

    Returns
    -------
    Jx,Jy,Jz : ndarray
        Body-fixed angular momentum components

    """
    
    NJ = 2*J+1
    
    kI, kJ = np.meshgrid(np.arange(NJ)-J, np.arange(NJ)-J, indexing = 'ij')
    
    Jz = np.zeros((NJ,NJ))
    
    Jz[kI==kJ] = kJ[kI==kJ]
    
    # Calculate Jp = Jx + i*Jy
    # (this is the *lowering* operating in body-fixed frame
    #  because of the anamalous commutation sign of Jx, Jy, Jz)
    Jp = np.zeros((NJ,NJ))
    Jm = np.zeros((NJ,NJ))
    
    idx = (kI == kJ - 1)
    Jp[idx] = np.sqrt( J*(J+1) - kJ*(kJ-1) )[idx] # lowering operator
    
    idx = (kI == kJ + 1)
    Jm[idx] = np.sqrt( J*(J+1) - kJ*(kJ+1) )[idx] # raising operator
    
    Jx = (Jp + Jm) / 2.0
    Jy = (Jp - Jm) / (2.0 * 1j)
    
    return Jx, Jy, Jz

def U_wr2cs(J):
    """
    Wang transformation matrix, with additional
    phase factors for real functions.
    
    cs = U @ wr

    Parameters
    ----------
    J : int
        Angular momentum quantum number, 0, 1, 2, ...

    Returns
    -------
    W : ndarray
        The unitary transformation matrix

    """
    
    NJ = 2*J+1
    
    U = np.zeros((NJ,NJ), dtype = np.complex128)
    
    def NJKplus(J,K):
        if J % 2 == 1 and K % 2 == 1:
            return -1j * (1j)**(J+1)
        elif J % 2 == 0 and K % 2 == 0:
            return +1j * (1j)**(J+1)
        else:
            return 1 * (1j)**(J+1)
    def NJKminus(J,K):
        if J % 2 == 1 and K % 2 == 0:
            return +1j * (1j)**(J+1)
        elif J % 2 == 0 and K % 2 == 1:
            return -1j * (1j)**(J+1)
        else:
            return 1 * (1j)**(J+1)
        
    
    for i in range(J):
        
        U[i,i] = -1.0 / np.sqrt(2.0) * NJKminus(J,J-i)
        U[-(i+1), i] = 1.0 / np.sqrt(2.0) * NJKminus(J,J-i)
        
        U[i, -(i+1)] = 1.0 / np.sqrt(2.0) * NJKplus(J,J-i)
        U[-(i+1), -(i+1)] = 1.0 / np.sqrt(2.0) * NJKplus(J,J-i)
    
    U[J,J] = 1.0

    return U

def iJbf_wr(J):
    """
    Calculate body-fixed J operators in real, symmetrized JK basis
    ("Wang-Real")

    Parameters
    ----------
    J : int
        Angular momentum quantum number, 0, 1, 2, ...

    Returns
    -------
    iJx, iJy, iJz : ndarray
        The body-frame angular momentum operators multiplied
        by i. This are purely real, anti-symmetric matrices

    """
    
    Jxyz = Jbf_cs(J) # Condon-Shortley representation
    
    U = U_wr2cs(J)
    Uh = U.conj().T
    
    iJx = np.real(1j * (Uh @ Jxyz[0] @ U))
    iJy = np.real(1j * (Uh @ Jxyz[1] @ U))
    iJz = np.real(1j * (Uh @ Jxyz[2] @ U))
    
    # Enforce some strict selection rules
    NJ = 2*J+1
    i,j = np.meshgrid(range(NJ),range(NJ),indexing = 'ij')
    
    iJx[abs(i-j) != 1] = 0  # Only the diagonal +/- 1 is non-zero
    iJy[abs(i-(NJ-j-1)) != 1] = 0 # Only the anti-diagonal +/- 1 is non-zero
    
    iJz[i != (NJ-j-1)] = 0 # Only the anti-diagonal is non-zero
    iJz[J,J] = 0
    
    # Let's enforce strict anti-symmetry
    for O in (iJx, iJy, iJz):
        for i in range(NJ):
            for j in range(i+1):
                if i == j:
                    O[i,j] = 0
                else:
                    O[j,i] = -O[i,j]
    
    return iJx, iJy, iJz
    
def iJiJbf_wr(J):
    """
    Calculate the anti-commutators [iJ_a,iJ_b]_+ for
    body-fixed angular momentum components in the Wang-real
    representation.

    Parameters
    ----------
    J : int
        Total angular momentum, 0, 1, 2, ...

    Returns
    -------
    iJiJ : nested tuple of ndarrays
        iJiJ[a][b] is the [iJa, iJb]_+ anti-commutator ndarray
    
    """
    
    iJ = iJbf_wr(J)
    
    def ac(A,B):
        return A@B + B@A
    
    iJiJ = tuple(tuple(ac(iJ[a],iJ[b]) for b in range(3)) for a in range(3))
    
    return iJiJ