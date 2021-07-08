"""
nitrogen.dvr.ops
----------------

DVR grid operations

"""

import numpy as np

from nitrogen.linalg.packed import IJ2k

def opDD_grid(x, f, h, G, Dlist):
    """
    Evaluate the matrix-vector operation for
    :math:`\sum_{k,\ell} f \partial_k^\dagger G^{kl} h \partial_\ell f`
    
    Parameters
    ----------
    x : ndarray
        Input grid
    f,h : ndarray
        Weighting functions evaluated over grid.
    G : ndarray
        Inverse metric grids in packed storage for active coordinates only.
    Dlist : list of ndarrays
        List of the D operator for each dimension. An entry of
        None will be skipped.

    Returns
    -------
    y : ndarray
        The result grid

    """
    
    N = len(Dlist) # The number of derivative operators
    
    # Check dimensions
    if np.ndim(G) != N + 1 or G.shape[1:] != x.shape:
        raise ValueError("G has an unexpected shape")
    if f.shape != x.shape:
        raise ValueError("f has an unexpected shape")
    if h.shape != x.shape:
        raise ValueError("h has an unexpected shape")
        
    
    
    fx = f * x
    
    y = np.zeros_like(x)
    #y.fill(0) # Initialize result array
    
    ellactive = 0
    for ell in range(N):
        Dl = Dlist[ell] # D_l
        if Dl is None:
            continue # do not increment ellactive
        dl = np.tensordot(Dl,fx,axes = (1,ell))
        dl = np.moveaxis(dl, 0, ell)
        hdx = h * dl # h * D_l * f * x
        
        kactive = 0
        for k in range(N):
            Dk = Dlist[k] # D_k
            if Dk is None:
                continue  # do not in increment kactive
            Ghdx = G[IJ2k(kactive,ellactive)] * hdx # G_kl * h * D_l * f * x
            dk = np.tensordot(Dk.T, Ghdx, axes = (1, k))
            dk = np.moveaxis(dk, 0, k)
            
            y += dk
            
            kactive += 1 
            
        ellactive += 1
    
    # Multiply by final weighting function f    
    y *= f
            
    return y

def opD_grid(x, f, Dlist):
    """
    Evaluate the matrix-vector operation for
    :math:`\sum_{k} -\partial_k^\dagger f_k + f_k \partial_k`
    
    Parameters
    ----------
    x : ndarray
        Input grid
    f : ndarray
        Weighting functions evaluated over grid for each
        *active* vibrational index `k`.
    Dlist : list of ndarrays
        List of the D operator for each dimension. An entry of
        None for inactive coordinates will be skipped.

    Returns
    -------
    y : ndarray
        The result grid

    """
    
    N = len(Dlist) # The number of derivative operators, including inactive
    
    # Check dimensions
    if np.ndim(f) != N + 1 or f.shape[1:] != x.shape:
        raise ValueError("f has an unexpected shape")
        
    y = np.zeros_like(x)
    #y.fill(0) # Initialize result array
    
    kactive = 0
    for k in range(N):
        Dk = Dlist[k] # D_k
        if Dk is None:
            continue  # do not increment kactive
        
        fk = f[kactive] # f grids are only provided for the active coords
        
        dk = np.tensordot(Dk, x, axes = (1,k)) # dk = D_k * x
        dk = np.moveaxis(dk, 0, k)
        y += fk * dk 
        
        fx = fk * x # f_k * x 
        dfx = np.tensordot(-Dk.T, fx, axes = (1,k))
        dfx = np.moveaxis(dfx, 0, k)
        y += dfx 

        kactive += 1
            
    return y

def opO_coeff(x, Olist, c = None):
    """
    Evaluate the matrix-vector operation for
    :math:`\sum_{k} c_k \hat{O}_k`
    
    Parameters
    ----------
    x : ndarray
        Input grid
    Olist : list of ndarrays
        List of the matrix operator for each dimension. An entry of
        None for inactive coordinates will be skipped.
    c : array_like, optional
        Coefficients for each term in sum, including inactive operators
        (whose values are ignored).  If None, this is ignored.

    Returns
    -------
    y : ndarray
        The result grid

    """
    
    N = len(Olist) # The number of operators, including inactive
        
    # Initialize result
    y = np.zeros_like(x)
    
    for k in range(N):
        Ok = Olist[k] # O_k
        if Ok is None:
            continue  # do not increment kactive
        
        dk = np.tensordot(Ok, x, axes = (1,k)) # dk = O_k * x
        dk = np.moveaxis(dk, 0, k)
        if c is not None:
            dk *= c[k] # Multiply coefficient
        y += dk 
        
            
    return y

def opO(x, O, axis):
    """
    Apply matrix operator O to a single axis
    
    Parameters
    ----------
    x : ndarray
        Input grid
    O : ndarray
        An (m,n) matrix
    axis : int
        The axis O acts on. x.shape[axis] must equal n.
        
    Returns
    -------
    y: ndarray
        The result array, with y.shape[axis] equal to m.

    """
    
    y = np.tensordot(O, x, axes = (1,axis) )
    y = np.moveaxis(y, 0, axis) 
    
    return y 