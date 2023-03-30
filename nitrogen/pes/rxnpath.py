"""
rxnpath.py

Reaction path routines
"""

import numpy as np 
import nitrogen


def qderiv_nonstationary(q0, deriv, V, cs, masses = None, mode = 'bodyframe',
                         direction = 'descend'):
    """
    Calculate the reaction path deriatives evaluated at an arbitrary
    (non-stationary) point.

    Parameters
    ----------
    q0 : array_like
        The evaluation point.
    deriv : integer
        The maximum derivative order to calculate.
    V : DFun
        The potential energy surface.
    cs : CoordSys
        The coordinate system.
    masses : array_like, optional
        The masses. If None, unit masses are assumed.
    mode : {'bodyframe'}, optional
        The frame type. The default is 'bodyframe'.
    direction: {'descend', 'ascend'}
        The direction of the path coordinate. If 'descend',
        then the path follows the negative gradient for increasing
        arc length.

    Returns
    -------
    q : (deriv + 1, nq)
        The derivative of the reaction path with respect to the
        arc length parameter.

    """
    
    q0 = np.array(q0)
    nvar = len(q0) # The number of coordinates
    
    
    # If deriv == 0, then only the value is
    # requested. Just return the evaluation point
    #
    if deriv == 0:
        q = np.array(q0).copy().reshape((1,nvar))
        return q 
    
    if direction == 'descend':
        sign = -1.0 
    elif direction == 'ascend':
        sign = +1.0 
    else:
        raise ValueError('direction must be either descend or ascend')
    
    # At least first derivatives are requested.
    # 
    # The PES needs to be evaluated to the same order
    # and the metric tensor (and inverse) to one
    # order less.
    #
    
    # nd_V = nitrogen.dfun.nderiv(deriv, nvar)
    nd_G = nitrogen.dfun.nderiv(deriv-1, nvar)
    
    # Calculate derivatives of the metric tensor, g
    g = cs.Q2g(q0, masses = masses, deriv = deriv - 1)
    
    # Calculate derivatives of the inverse metric, G
    G,_ = nitrogen.dfun.sym2invdet(g, deriv - 1, nvar)
    
    # Extract just the vibrational block
    dG = np.zeros((nd_G, nvar, nvar))
    
    if mode == 'bodyframe':
        #
        # Calculate all coordinate derivatives of 
        # Gvib 
        idx = 0
        for i in range(nvar):
            for j in range(i+1):
                np.copyto(dG[:,i,j], G[:,idx]) 
                if i != j:
                    np.copyto(dG[:,j,i], G[:,idx]) 
                idx = idx + 1 
    else:
        raise ValueError(f'Unrecognized mode = {str(mode):s}')
    
    # 
    # Calculate the derivatives of the PES
    # arranged as derivatives of the gradient
    #
    df = V.jacderiv(q0, deriv - 1)[:,:,0] # (nd, nvar, 1, ...)
    
 
    # Now begin the recursive evaluation 
    # of the path derivatives
    #
    
    #############################
    # Initialize the first derivative, q1 = q^(1)
    # 
    # q^(1) = dq/ds = v = +/- Gvib . f / c =  +/- F / c 
    # c = sqrt[F.f]

    
    f0 = df[0] # f_vib 
    G0 = dG[0] # G_vib 
    F0 = G0 @ f0  # The covariant gradient
    c0 = np.sqrt(F0 @ f0)  
    v0 = sign * F0 / c0 # Sign depends on ascent or descent 
    q1 = v0
    
    qn = [q0,q1]
    Gn = [G0]
    fn = [f0]
    Fn = [F0]
    cn = [c0] 
    vn = [v0] 
    
    ##########################
    #
    for n in range(1,deriv):
        # 
        # Calculate q(n + 1)
        #
        # qn has [q0,q1,...,qn]
        # Gn has [G0,...G(n-1)]
        # fn has [f0,...f(n-1)]
        #
        # Add G(n) and f(n) to lists
        # calculated by chain rule
        Gn.append(pathderivchain(dG, np.stack(qn, axis = 0))[-1])
        fn.append(pathderivchain(df, np.stack(qn, axis = 0))[-1])
        
        # Calculate F(n)
        # 
        # F = G @ f 
        #
        Fn.append(sum([Gn[m] @ fn[n-m] for m in range(n+1)])) # m = 0...n
       
        
        # Calculate c(n)
        #
        # c**2 = F @ f 
        #
        temp = 0 
        for m in range(0,n+1): # m = 0...n
            temp = temp + np.dot(Fn[m],fn[n-m])
            if m > 0 and m < n:
                temp = temp - cn[m] * cn[n-m]  # m = 1...n-1
        cn.append(0.5 * temp / cn[0]) 
        
        # Calculate v(n)
        # 
        # v = sign * F/c
        temp = 0 
        temp = temp + sign * Fn[n] 
        for m in range(1,n+1): # m = 1 ... n
            temp = temp - cn[m] * vn[n-m] 
        vn.append(temp / cn[0])
        
        # Calculate q(n+1)
        #
        # q(n+1) = 1/(n+1) * v(n)
        qn.append( vn[n] / (n+1) )
        
    #
    # qn contains [q0,q1,...,q(deriv)]
    #
    qn = np.stack(qn, axis = 0) # Stack into a derivative array 
    
    return qn 

def qderiv_stationary(q0, deriv, V, cs, masses = None, mode = 'bodyframe',
                      direction = 'normal'):
    """
    Calculate the reaction path deriatives evaluated at a stationary
    point.

    Parameters
    ----------
    q0 : array_like
        The evaluation point.
    deriv : integer
        The maximum derivative order to calculate.
    V : DFun
        The potential energy surface.
    cs : CoordSys
        The coordinate system.
    masses : array_like, optional
        The masses. If None, unit masses are assumed.
    mode : {'bodyframe'}, optional
        The frame type. The default is 'bodyframe'.
    direction: {'normal', 'reverse'}, optional
        The direction of the path coordinate. If 'normal',
        the sign of the path tangent is determined by making
        its largest element positive. If 'reverse', the sign is 
        reversed.

    Returns
    -------
    q : (deriv + 1, nq) ndarray 
        The derivative of the reaction path with respect to the
        arc length parameter.

    """
    
    q0 = np.array(q0)
    nvar = len(q0) # The number of coordinates
    
    
    # If deriv == 0, then only the value is
    # requested. Just return the evaluation point
    #
    if deriv == 0:
        q = np.array(q0).copy().reshape((1,nvar))
        return q 
        
    # nd_V = n2.dfun.nderiv(deriv+1, nvar) # extra derivative order needed
    nd_G = nitrogen.dfun.nderiv(deriv-1, nvar)
    
    # Calculate deriatives of metric tensor, g
    g = cs.Q2g(q0, masses = masses, deriv = deriv - 1)
    
    # Calculate derivatives of the inverse metric
    G,_ = nitrogen.dfun.sym2invdet(g, deriv - 1, nvar)
    
    
    dG = np.zeros((nd_G, nvar, nvar))
    
    if mode == 'bodyframe':
        #
        # Extract vibrational block of G
        
        idx = 0
        for i in range(nvar):
            for j in range(i+1):
                np.copyto(dG[:,i,j], G[:,idx]) 
                if i != j:
                    np.copyto(dG[:,j,i], G[:,idx]) 
                
                idx = idx + 1 
    else:
        raise ValueError(f'Unrecognized mode = {str(mode):s}')
    # 
    # Calculate derivatives of the PES
    # organized as derivatives of the Hessian
    #
    dK = V.hesderiv(q0, deriv - 1)[:,:,:,0] # (nd, nvar, nvar, ...)
    
    #############################
    # Calculate the first path derivative
    # q1 = q^(1) = v(0)
    # 
    # v(0) is the eigenvector 
    # of GK with lowest eigenvalue
    #
    G0 = dG[0]   # G_vib 
    K0 = dK[0]   # K_0, the hessian 
    
    # Calculate eigenvector
    w,U = np.linalg.eig(G0 @ K0)
    I = np.argsort(w) # Sort by eigenvalue
    w = w[I] 
    U = U[:,I] 
    # Take the lowest-eigenvalue solution
    v0 = U[:,0] # unnormalized 
    lam = w[0] # The eigenvalue 
    #
    # If the eigenvalue is negative, then this is 
    # steepest descent from a transition state
    #
    # If the eigenvalue is positive, then this is
    # steepest descent from a local minimum
    #
    
    # The normalization condition is v @ G^-1 @ v = 1.
    # Because v is an eigenvector of GK, we have
    # 
    # v @ G^-1 @ v = v @ K @ v / lambda = 1
    # --> v @ K @ v = lambda
    #
    v0 = v0 * np.sqrt( lam / (v0.T @ K0 @ v0) ) # normalize v0 
    #
    # Make an arbitrary sign choice of the initial direction
    # Choose the sign such that the largest element of v0 is
    # positive for direction = 'normal', and reverse this if
    # direction = 'reverse'
    imax = np.argmax(abs(v0))
    if v0[imax] < 0:
        v0 = -v0 
        
    if direction == 'reverse':
        v0 = -v0 
    
    #
    # Initialize the zeroth derivatives
    #
    f0 = 0*q0 # The gradient is zero.
    F0 = 0*q0 # F = G @ f is zero at the stationary point.
    c0 = 0.0  # c = F @ f is zero
    q1 = v0   # The path tangent
    
    qn = [q0,q1]
    Kn = [K0] 
    Gn = [G0]
    fn = [f0]
    Fn = [F0]
    cn = [c0] 
    vn = [v0] 
    
    # F(1), f(1), and c(1)
    # will be needed explicitly, so let's just calculate them 
    # now
    #
    # f(1) = K(0) @ v(0)
    f1 = K0 @ v0 
    # 
    # F(1) = G(1) @ f(0) + G(0) @ f(1) 
    #      = G(0) @ f(1) at stationary point (where f(0) = 0)
    F1 = G0 @ f1 
    
    #
    # dc/ds at stationary point = |v0.T @ K0 @ v0| = |lambda|
    # (Note that c1 must always be positive regardless of the 
    #  sign of lambda)
    c1 = abs(lam)
    
    ##########################
    #
    for n in range(1,deriv):
        # 
        # Calculate q(n + 1)
        #
        # qn has [q0,q1,...,qn]
        # Gn has [G0,...G(n-1)]
        # Kn has [K0,...K(n-1)]
        # Add G(n) and K(n) to lists
        Gn.append(pathderivchain(dG, np.stack(qn, axis = 0))[-1])
        Kn.append(pathderivchain(dK, np.stack(qn, axis = 0))[-1])
        
        # Calculate f(n)
        # 
        # df/ds = K @ v 
        #
        # f(n) = 1/n * (K @ v)^(n-1)
        fn.append(sum([Kn[m] @ vn[n-1-m] for m in range(n)]) / n) # m = 0...n-1
        
        # Calculate F(n)
        # 
        # F = Gvib @ f 
        #
        Fn.append(sum([Gn[m] @ fn[n-m] for m in range(n+1)])) # m = 0 ... n
        
        # Calculate c(n)
        #
        # (c/s)**2 = F/s @ f/s
        #
        factor = 0.5 
        if n == 1:
            factor = 1.0 
            
        cn.append( factor/c1  * (
            sum([Fn[n-m] @ fn[m+1] for m in range(n)]) +  # m = 0...n-1
            sum([cn[n-m] * cn[m+1] for m in range(1,n-1)]) # m = 1...n-2
            ))
        
        #######################
        # Now calculate the components 
        # needed for the v(n) linear equation
        An = Kn[0]/(n+1) 
        an = sum([Kn[n-m] @ vn[m] for m in range(n)])  # m = 0...n-1
        an = an / (n+1) 
        
        Bn = G0 @ An 
        bn = G0 @ an + sum([Gn[n-m] @ fn[m+1] for m in range(n)]) # m = 0...n-1
         
        DnT = 0.5 * (F1.T @ An + f1.T @ Bn)
        dn = 0.5 * (F1.T @ an + f1.T @ bn + sum(
            [Fn[m+1].T @ fn[n+1-m] - cn[m+1]*cn[n+1-m] for m in range(1,n) ]  ) ) # m = 1...n-1
        
        #
        # Note the presence of lambda in the following
        # equations. This accounts for the descent/ascent
        # character of the path tangent.
        #
        Tn = -lam * Bn + np.outer(v0, DnT) + (lam**2) * np.eye(len(an))
        tn = -lam * bn + dn * v0 + cn[1] * sum([cn[n+1-m] * vn[m] for m in range(1,n)]) # m = 1...n-1
        
        # v(n) satisfies
        # 
        #  Tn @ v(n) = - tn 
        #
        vn.append(np.linalg.solve(Tn, -tn))
        
        # Calculate q(n+1)
        #
        # q(n+1) = 1/(n+1) * v(n)
        qn.append( vn[n] / (n+1) )
        
    # End loop.
    # qn contains [q0,q1,...,q(deriv)]
    
    qn = np.stack(qn, axis = 0) # Stack into a derivative array 
    
    return qn 
    
def pathderivchain(A,X):
    """
    Calculate the derivative of a quantity with respect to the
    path parameter via the multivariate chain rule.
    
    Parameters
    ----------
    A : (nd,...) ndarray
        The derivative array of :math:`A(x)` 
        with respect to `nvar` :math:`x` coordinates.
    X : (deriv+1,nvar) ndarray
        The derivative array of `nvar` `:math:`x` coordinates
        with respect to the path parameter :math:`s`.
    
    Returns
    -------
    B : (deriv+1,...) ndarray
        The derivative array of :math:`B(s) = A(x(s))`
        
    """
       
    ########################################
    #
    def singleLeibniz(a1,a2):
        """
        A simple single-variable Leibniz product rule
        
        a1,a2 : (deriv+1,...) ndarray
            Single variable deriative arrays
        
        """
        result = np.zeros_like(a1)
        deriv = a1.shape[0] - 1
        for i in range(deriv+1):
            for j in range(deriv + 1):
                k = i + j # total order 
                if k > deriv:
                    continue 
                result[k] += a1[i] * a2[j] 
        return result 
    #
    #########################################
    
    #########################################
    # Perform a many-to-single chain rule
    #
    
    deriv = X.shape[0] - 1 # The derivative order 
    nvar = X.shape[1] # The number of variables 
    base_shape = A.shape[1:]
    
    # The derivatives of A w.r.t `s` with be calculated
    # with the Taylor series chain rule 
    #
    # We first need the derivative arrays of the powers 
    # of each `x` coordinate w.r.t `s` 
    #
    deltaX = X.copy() 
    deltaX[0] = 0.0 
    # deltaX is the derivative array w.r.t. s 
    # for the displacement of the coordinates
    
    xpow = np.zeros((deriv + 1, nvar, deriv + 1))
    
    # xpow[:,k,p] is the derivative array for the
    # p**th power of (delta x)_k
    #
    if deriv >= 0:
        xpow[0:1, :, 0] = 1.0 # Zeroth power = 1.0 constant
    if deriv >= 1:
        np.copyto(xpow[:, :, 1], deltaX[:,:]) # First power = coordinate displacement
    for p in range(2,deriv+1):
        # higher powers
        xp = singleLeibniz(xpow[:,:,p-1], xpow[:,:,1])
        np.copyto(xpow[:, :, p], xp)
    
    # Now that we have the derivative arrays for the powers of 
    # the coordinates, we can construct the derivative array
    # of A w.r.t s via its truncated Taylor series 

    idxtab = nitrogen.autodiff.forward.idxtab(deriv, nvar) 
    result = np.zeros( (deriv+1,) + base_shape )
    
    one = np.zeros((deriv+1,))
    one[0] = 1.0 
    # one is the `s` derivative array for 1
    
    for idxA in range(idxtab.shape[0]):
        p = idxtab[idxA] # The multi index of this term of A
        
        temp = one # temp <-- 1.0 
        for k in range(nvar):
            temp = singleLeibniz(temp, xpow[:, k, p[k]])
        # temp now equals the product of powers of 
        # x-displacements 
        #
        temp = temp.reshape((deriv+1,) + tuple([1]*len(base_shape)))
        result += temp * A[idxA]
    
    return result 
   