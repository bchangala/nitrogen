# -*- coding: utf-8 -*-
"""
opt.py

Geometry optimizer routines
"""
import numpy as np 

__all__ = ['opt_newton', 'opt_bfgs']



def opt_newton(F, x0, alpha = 1.0, tol = 1e-6, disp = False,
               fidx = 0, var = None, maxiter = 100):
    """
    Simple Newton-Raphson optimization.

    Parameters
    ----------
    F : DFun
        A DFun object with second derivative support.
    x0 : array_like
        The initial coordinate values.
    alpha : float, optional
        The update step scaling coefficint. The default is 1.0.
    tol : float, optional
        The gradient norm tolerance. The default is 1e-6.
    disp : bool, optional
        Print detailed output. The default is False.
    fidx : int, optional
        The DFun function index to optimize. The default is 0.
    var : list of int, optional
        The variables to optimize. All others will be constrained
        to the values in `x0`. If None (default), all coordinates
        will be optimized
    maxiter : int, optional
        The maximum number of steps. The default is 100.

    Returns
    -------
    
    xopt : ndarray
        The optimized coordinates
    Fopt : scalar
        The value of the optimized function 

    """
    
    if var is None:
        var = [i for i in range(len(x0))]
        
    ntot = len(x0)  # The total number of coordinates
        
    if F.nx != ntot: 
        raise ValueError("F and x0 are inconsistent")
        
    x0 = np.array(x0) # The initial geometry
    xi = x0.copy()    # The present geometry 
        
    cnt = 0
    conv = False 
    
    if disp:
        print("Step   Value       |grad|")
        print("-----------------------------")
    for i in range(maxiter):
        #
        # Simple Newton-Raphson
        # 
        # 1) Calculate the local gradient and Hessian 
        vjh = F.vjh(xi, var = var); cnt = cnt + 1 
        
        v = vjh[0][fidx] # The value
        g = vjh[1][fidx] # The gradient w.r.t. var
        H = vjh[2][fidx] # The Hessian w.r.t. var 
        
        # 2) Check convergence
        gnorm = np.linalg.norm(g)
        
        if disp:
            print(f" {cnt:2d}    {v:.4e}  {gnorm:.4e}  ...  ", end = "")
            print(xi[var])
        
        if gnorm <= tol:
            conv = True 
            break 
        
        # 3) Calculation the displacment of var
        dx = -alpha * np.linalg.inv(H) @ g
        
        # 4) Update the geometry
        xi[var] += dx 
        
    if disp:
        print("")
        if conv:
            print(f"Convergence reached, |g| = {gnorm:.3e}")
        else:
            print(f"Maximum iterations reached, |g| = {gnorm:.3e}")
        print(f"{cnt:d} gradient(s) and {cnt:d} Hessian(s) were calculated.")
        
    Fopt = v
    xopt = xi  # Return the total geometry
    
    return xopt, Fopt 

def opt_bfgs(F, x0, Hinit = None, alpha = 1.0, tol = 1e-6, disp = False,
               fidx = 0, var = None, maxiter = 100):
    """
    
    Quasi-Newton optimization with a Broyden-Fletcher-Goldfarb-Shannon 
    (BFGS) Hessian update.    

    Parameters
    ----------
    F : DFun
        A DFun object with second derivative support.
    x0 : array_like
        The initial coordinate values.
    Hinit : array_like, optional
        The initial Hamiltonian. If None, the initial Hessian will be 
        calculated explicitly. 
        If array_like, then a user-supplied array is used. Note that
        `Hinit` must be supplied with respect to only the variables
        in `var` and in that order.
    alpha : float, optional
        The update step scaling coefficint. The default is 1.0.
    tol : float, optional
        The gradient norm tolerance. The default is 1e-6.
    disp : bool, optional
        Print detailed output. The default is False.
    fidx : int, optional
        The DFun function index to optimize. The default is 0.
    var : list of int, optional
        The variables to optimize. All others will be constrained
        to the values in `x0`. If None (default), all coordinates
        will be optimized
    maxiter : int, optional
        The maximum number of steps. The default is 100.
        
    Returns
    -------
    
    xopt : ndarray
        The optimized coordinates
    Fopt : scalar
        The value of the optimized function 

    """

    if var is None:
        var = [i for i in range(len(x0))]
        
    ntot = len(x0)  # The total number of coordinates
        
    if F.nx != ntot: 
        raise ValueError("F and x0 are inconsistent")
        
    x0 = np.array(x0) # The initial geometry
    xi = x0.copy()    # The present geometry 
        
    cnt1 = 0 # Value/Jacobian count
    cnt2 = 0 # Hessian count
    
    # Process initial Hamiltonian
    if Hinit is None:
        # Calculate the Hessian explic
        cnt2 += 1 
        Hinit = F.hes(xi, var = var)[fidx] 
    else: # assume array_like
        Hinit = np.array(Hinit) 
    
    conv = False     

    if disp:
        print("Step   Value       |grad|")
        print("-----------------------------")
        
    
    Hnew = Hinit
    gold = 0 
    Hold = 0 
    dx = 0 
    for i in range(maxiter):
        #
        #
        # BFGS Hessian update
        #
        # 1) Calculate the local gradient
        
        vj = F.vj(xi, var = var); cnt1 = cnt1 + 1 
        
        v = vj[0][fidx] # The value
        g = vj[1][fidx] # The gradient w.r.t. var
        
        # 2) Check convergence
        gnorm = np.linalg.norm(g)
        
        if disp:
            print(f" {i+1:2d}    {v:.4e}  {gnorm:.4e}  ...  ", end = "")
            print(xi[var])
        
        if gnorm <= tol:
            conv = True 
            break 
        
        # 3) Update Hessian using previous step
        if i > 0:
            dg = g - gold # The gradient difference
            # dx already equals the coordinate difference 
            Hdx = Hold @ dx 
            dH = np.outer(dg,dg)/np.dot(dg,dx) - np.outer(Hdx,Hdx) / np.dot(dx, Hdx)
            Hnew = Hold + dH 
            
        #
        # 4) Calculate displacement
        #    using the updated Hessian 
        dx = -alpha * np.linalg.inv(Hnew) @ g 
        
        # 4) Update the geometry
        xi[var] += dx 
        
        Hold = Hnew 
        gold = g 
    #
    ##########################

    if disp:
        print("")
        if conv:
            print(f"Convergence reached, |g| = {gnorm:.3e}")
        else:
            print(f"Maximum iterations reached, |g| = {gnorm:.3e}")
        print(f"{cnt1:d} gradient(s) and {cnt2:d} Hessian(s) were calculated.")
            
    Fopt = v
    xopt = xi  # Return the total geometry
    
    return xopt, Fopt 

