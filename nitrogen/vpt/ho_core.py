"""
Core harmonic oscillator algebra functions
"""

import numpy as np 

def a_ap_matrix(n):
    """
    Calculate the matrix representations 
    of the annihilation and creation
    operators, :math:`a` and :math:`a^\\dagger`.

    Parameters
    ----------
    n : int
        The number of states. The last state is
        :math:`\\vert n-1 \\rangle`.

    Returns
    -------
    a : ndarray
        The (`n`,`n`) matrix representation of 
        :math:`a`.
    ap : ndarray
        The (`n`,`n`) matrix representation of
        :math:`a^\\dagger`.

    """
    
    a = np.zeros((n,n))
    ap = np.zeros((n,n))
    
    for i in range(n-1):
        a[i,i+1] = np.sqrt(i+1)
        ap[i+1,i] = np.sqrt(i+1)
    
    
    return a,ap 
    
def q_matrix(n, m = 1):
    """
    Calculate the matrix representation 
    of :math:`q^m`, where :math:`q = \\sqrt{\\frac{1}{2}}(a + a^\\dagger)`.

    Parameters
    ----------
    n : int
        The number of states. The last state is
        :math:`\\vert n-1 \\rangle`.
    power : int, optional
        The power of :math:`q`. The default is 1. Must be 
        a non-negative integer.

    Returns
    -------
    q : ndarray
        The (`n`,`n`) matrix representation of :math:`q^m`.

    """
    
    if m < 0 :
        raise ValueError('m must be a non-negative integer')
    
    if m == 0:
        q = np.eye(n)
    else:
        # m is a positive integer
        
        ntemp = n + (m - 1) # Use a large enough temp matrix size
                            
        a,ap = a_ap_matrix(ntemp)
        qtemp = np.sqrt(0.5) * (a + ap) 
        
        q = qtemp
        for i in range(m-1):
            q = q @ qtemp
            
        q = q[:n,:n].copy()
        
    return q

def p_matrix(n, m = 1):
    """
    Calculate the matrix representation 
    of :math:`p^m`, where :math:`p = -i \\sqrt{\\frac{1}{2}}(a - a^\\dagger)`.

    Parameters
    ----------
    n : int
        The number of states. The last state is
        :math:`\\vert n-1 \\rangle`.
    power : int, optional
        The power of :math:`p`. The default is 1. Must be 
        a non-negative integer.

    Returns
    -------
    p : ndarray
        The (`n`,`n`) matrix representation of :math:`p^m`.

    """
    
    if m < 0 :
        raise ValueError('m must be a non-negative integer')
    
    if m == 0:
        p = np.eye(n)
    else:
        # m is a positive integer
        
        ntemp = n + (m - 1) # Use a large enough temp matrix size
                            
        a,ap = a_ap_matrix(ntemp)
        ptemp =  -1j * np.sqrt(0.5) * (a - ap) 
        
        p = ptemp
        for i in range(m-1):
            p = p @ ptemp
            
        p = p[:n,:n].copy()
        
    return p