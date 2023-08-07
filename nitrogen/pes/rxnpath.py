"""
rxnpath.py

Reaction path routines
"""

import numpy as np 
import nitrogen


def qderiv_nonstationary(q0, deriv, V, G, direction = 'descend'):
    """
    Calculate the reaction path deriatives evaluated at an arbitrary
    (non-stationary) point.

    Parameters
    ----------
    q0 : (nq,...) array_like
        The evaluation points
    deriv : integer
        The maximum derivative order to calculate.
    V : DFun
        The potential energy surface.
    G : DFun
        The inverse metric tensor.
    direction: {'descend', 'ascend'}
        The direction of the path coordinate. If 'descend',
        then the path follows the negative gradient for increasing
        arc length.

    Returns
    -------
    q : (deriv + 1, nq, ...)
        The derivatives of the reaction path with respect to the
        arc length parameter.

    """
    
    q0 = np.array(q0)
    nvar = q0.shape[0] # The number of coordinates
    base_shape = q0.shape[1:] 
    
    # If deriv == 0, then only the value is
    # requested. Just return the evaluation points
    #
    if deriv == 0:
        q = np.array(q0).copy().reshape((1,nvar) + base_shape)
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
    if G is None:
        # Identity
        nd = nitrogen.dfun.nderiv(deriv-1, nvar)
        dG = np.zeros((nd,nvar,nvar))
        for i in range(nvar):
            dG[0,i,i] = 1.0    
    else:
        dG = symfull_axis(G.f(q0, deriv - 1), axis = 1) # Reshape

    
    # 
    # Calculate the derivatives of the PES
    # arranged as derivatives of the gradient
    #
    df = V.jacderiv(q0, deriv - 1)[:,:,0] # (nd, nvar, ...)
    
 
    # Now begin the recursive evaluation 
    # of the path derivatives
    #
    
    #############################
    # Initialize the first derivative, q1 = q^(1)
    # 
    # q^(1) = dq/ds = v = +/- Gvib . f / c =  +/- F / c 
    # c = sqrt[F.f]

    def matvec(M,v):
        # 
        # Matrix vector product with shapes
        # (n,m,...) @ (m,...) --> (n,...)
        #
        # Multiply via broadcasting and then sum over the second index
        #
        return np.sum(M*v, axis = 1)
    
    def dot(v,w):
        #
        # Vector dot product with shapes
        # (n,...) . (n,...) --> (...)
        #
        return np.sum(v*w, axis = 0)
        
    f0 = df[0] # f_vib 
    G0 = dG[0] # G_vib 
    F0 = matvec(G0,f0) # The covariant gradient
    c0 = np.sqrt(dot(F0,f0))  
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
        Fn.append(sum([matvec(Gn[m],fn[n-m]) for m in range(n+1)])) # m = 0...n
       
        
        # Calculate c(n)
        #
        # c**2 = F @ f 
        #
        temp = 0 
        for m in range(0,n+1): # m = 0...n
            temp = temp + dot(Fn[m],fn[n-m])
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

def qderiv_stationary(q0, deriv, V, G, direction = 'normal'):
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
    G : DFun
        The inverse metric tensor. If None, identity is assumed
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
        
    if G is None:
        # Identity
        nd = nitrogen.dfun.nderiv(deriv-1, nvar)
        dG = np.zeros((nd,nvar,nvar))
        for i in range(nvar):
            dG[0,i,i] = 1.0    
    else:
        dG = symfull_axis(G.f(q0, deriv - 1), axis = 1) # Reshape
    
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

def LQA_nonstationary(q0, V, G, arclength = 'massweighted',
                      proxy_index = 0):
    """
    Compute the covariant gradient (i.e., the reaction path tangent
    vector) within a local quadratic approximation at 
    non-stationary points.

    Parameters
    ----------
    q0 : (nq,...) array_like 
        The non-stationary evaluation points.
    V : DFun
        The potential energy surface 
    G : DFun
        The inverse metric. If None, a constant unit metric is assumed.
    arclength : {'massweighted','gradient','proxy'}, optional
        The path parameterization convention. The default is 'massweighted'.
    proxy_index : integer, optional
        The proxy coordinate index.

    Returns
    -------
    w0 : (nq,...) ndarray
        The path tangent at `q0`.
    W : (nq,nq,...) ndarray
        The path tangent-gradient at `q0`. ``W[i,j]`` is equal to 
        :math:`\\partial_j w_i \\vert_0`.
    G0 : (nq,nq,...) ndarray
        The inverse metric evaluated at `q`. The inner product of 
        tangent vectors (**not** gradient vectors) with `G0` yields
        their proper 2-norm.
    
    Notes
    -----
    
    The path tangent is defined as 
    
    ..  math::
        
        \\mathbf{w} = -G\\mathbf{f}/h,
                    
    where :math:`\\mathbf{f}` is the gradient, :math:`G` is the inverse metric
    tensor, and :math:`h` defines the path parameterization normalization.
    
    The LQA approximates the local path tangent as 
    
    ..  math::
        
        \\mathbf{w} \\approx \\mathbf{w}_0 + W(\\mathbf{q} - \\mathbf{q}_0)
    
    For `arclength` == ``'massweighted'``, the natural mass-weighted arc length
    is used, :math:`h = (\\mathbf{f}^T G \\mathbf{f})^{1/2}`.
    
    For `arclength` == ``'gradient'``, :math:`h = 1`.
    
    For `arclength` == ``'proxy'``, :math:`h = -(G\\mathbf{f})_i`, where :math:`i` is the
    coordinate index specified by `proxy_index`. In this case, the path is parameterized
    by one of the coordinates themselves -- the ''proxy'' coordinate -- so :math:`w_i = 1`.
    
    """
    
    # Calculate the local quadratic approximation to the 
    # path tangent
    #
    # Let w = -F/h
    #     F = G @ f
    #
    # where `f` is the gradient and
    # `h` defines the normalization convention.
    # 
    #
    q0 = np.array(q0)
    
    # Calculate the inverse metric tensor and its first derivatives
    #
    dG = symfull_axis(G.f(q0, deriv = 1), 1) 
    
    G0 = dG[0]   # G_ij (...)
    G1 = dG[1:]  # d_i G_jk (...)
    
    # Calculate the energy gradient and hessian
    _,f,K = [a[0] for a in V.vjh(q0)]
    # f : (nq,...) the local gradient
    # K : (nq,nq,...) the local Hessian
    
    #
    # Calculate F = G @ f and its derivatives
    #
    F0 = np.einsum('ij...,j...->i...', G0, f) # G0 @ f : (nq,...)
    F1 = np.einsum('ijk...,k...->ij...',G1,f) + np.einsum('jk...,ik...->ij...',G0,K)
    # F1 : (nq,nq,...)
    # F1[i,j] = d_i F_j
    
    #
    # Calculate the normalization parameter `h` and its derivatives
    if arclength == 'massweighted':
        #
        # h = sqrt[f.T @ G @ f]
        #   = sqrt[f.F]
        h0 = np.sqrt(np.sum(f*F0,axis=0)) 
        h1 = (np.einsum('j...,ij...->i...', f,F1) + 
              np.einsum('ij...,j...->i...', K,F0)) / (2*h0) 
        
    elif arclength == 'gradient': 
        # No normalization
        #
        h0 = 1.0 
        h1 = 0.0 
        
    elif arclength == 'proxy':
        #
        # h = -F*
        #
        h0 = -F0[proxy_index] 
        h1 = -F1[:,proxy_index]
        
    else: 
        raise ValueError('Invalid arclength mode = {str(arclength):s}')

    # Calculate the path tangent and its derivatives
    #
    # w = -F/h
    #
    #      h*w = -F 
    # --> dh*w + h*dw = -dF 
    #              dw = -(dF + dh*w) / dh
    w0 = -F0 / h0 
    w1 = -(F1 + np.einsum('i...,j...->ij...',h1,w0)) / h0 
    # w1[i,j] = d_i w_j
    
    #
    # The local tangent is approximated as 
    # 
    # w  ~  w0 + W@(q-q0)
    # 
    # so W equals the transpose of `w1`
    #
    W = np.einsum('ij...->ji...',w1) 
    
    return w0, W, G0
    
    
    
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
   
def pathderivchain(A,X):
    """
    Calculate the derivative of a quantity with respect to the
    path parameter via the multivariate chain rule.
    
    Parameters
    ----------
    A : (nd,...',...) ndarray
        The derivative array of :math:`A(x)` 
        with respect to `nvar` :math:`x` coordinates.
    X : (deriv+1,nvar,...) ndarray
        The derivative array of `nvar` `:math:`x` coordinates
        with respect to the path parameter :math:`s`.
    
    Returns
    -------
    B : (deriv+1,...',...) ndarray
        The derivative array of :math:`B(s) = A(x(s))`
        
    """
       

    
    #########################################
    # Perform a many-to-single chain rule
    #
    
    deriv = X.shape[0] - 1 # The derivative order 
    nvar = X.shape[1] # The number of variables 
    
    Ashape = A.shape[1:] 
    Xshape = X.shape[2:]
    Aax = len(Ashape) - len(Xshape) # The number of additional A axes
    
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
    
    xpow = np.zeros((deriv + 1, nvar, deriv + 1) + Xshape)
    
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
    result = np.zeros( (deriv+1,) + Ashape )
    
    one = np.zeros((deriv+1,) + Xshape)
    one[0:1].fill(1.0) 
    # one is the `s` derivative array for 1
    
    for idxA in range(idxtab.shape[0]):
        p = idxtab[idxA] # The multi index of this term of A
        
        temp = one # temp <-- 1.0 
        for k in range(nvar):
            temp = singleLeibniz(temp, xpow[:, k, p[k]])
        # temp now equals the product of powers of 
        # x-displacements 
        #
        temp = np.expand_dims(temp, tuple(range(1,Aax+1)))
        
        # temp =   (deriv+1, 1,...'1,  ... )
        # A[idx] =          (  ...' ,  ... )
        #
        #     B    (deriv+1,   ...' ,  ... )
        result += temp * A[idxA]
    
    return result 

def invertderiv(f, x0 = 0.0):
    
    """
    Calculate the derivatives of the inverse function
    given the derivatives of a function.
    
    Parameters
    ----------
    f : (deriv+1,...) ndarray
        The derivative array of the original function, :math:`f(x)`.
    x0 : (...) ndarray or scalar, optional
        The expansion value of :math:`x`.  The default is zero.
    
    Returns
    -------
    F : (deriv+1,...) ndarray
        The deriative array of the inverse function, :math:`F(y)`.
    
    
    """
    
    # We will take advantage of the Taylor series/chain rule
    # operation to implicitly perform the inverse recursion 
    # process
    
    deriv = f.shape[0] - 1 # The derivative order 
    base_shape = f.shape[1:] 
    
    F = np.zeros_like(f) 
    
    F[0] = x0  # The value of the inverse function is just the expansion point 
    
    # Construct the derivative array
    # of the displacement of `f`
    df = f.copy() 
    df[0] = 0.0  # Value is 0.0 at expansion
    
    # And calculate its powers 
    dfpow = np.zeros((deriv + 1, deriv + 1) + base_shape)
    
    # dfpow[:,p] is the derivative array for the
    # p**th power of the displacement of f
    #
    if deriv >= 0:
        dfpow[0:1, 0] = 1.0 # The zeroth power is just 1.0 constant
    if deriv >= 1:
        # The first power is the displacement of f
        np.copyto(dfpow[:, 1], df) # First power = coordinate displacement
    for p in range(2,deriv+1):
        # higher powers
        df_p = singleLeibniz(dfpow[:,p-1], dfpow[:,1]) 
        np.copyto(dfpow[:, p], df_p)
        
    #
    # Now the derivatives of F must satisfy
    #
    # F(0) * [df]**0 + 
    # F(1) * [df]**1 + 
    # F(2) * [df]**2 + ...  =  F
    #
    # The derivative array of F w.r.t itself is just (F0, 1, 0, 0, 0, ...)
    #
    # i.e. F(0) = x0
    #      F(1) * df(1) = 1 --> F(1) = 1 / df(1)
    F[1] = 1.0 / df[1] # note f[1] = df[1] 
    
    for n in range(2, deriv+1):
        
        temp = sum([F[m] * dfpow[n,m] for m in range(1,n)]) # m = 1 ... n-1
        F[n] = -temp / dfpow[n,n]
    
    return F 

def proxyderiv(qn, star_index):
    """
    Convert path derivatives to those with respect to 
    one of the coordinates
    
    Parameters
    ----------
    qn : (deriv+1,n,...) 
        The derivative array of the `n` coordinates with respect to 
        a path parameter
    star_index : integer
        The index of the proxy coordinate
    
    Returns
    -------
    qn_star : (deriv+1,n,...)
        The derivatives of the `n` coordinates with respect to the
        proxy coordinate
    
    """
    
    # Extract the derivatives of q* w.r.t the path paramter `s`
    qn = np.array(qn) # (nd,n,...)
    
    qstar_wrt_s = qn[:,star_index] # (nd,...)
    
    # Now invert the derivatives to get `s` w.r.t q*
    s_wrt_qstar = invertderiv(qstar_wrt_s) # (nd,...)
    
    # Now use the chain rule to compute other coordinates w.r.t. q* 
    qn_star = np.zeros_like(qn)
    for i in range(qn.shape[1]): # for each coordinate
        qn_star[:,i] = pathderivchain(qn[:,i], s_wrt_qstar[:,np.newaxis,...] ) 
    
    # Note that the derivative array for q* w.r.t q* itself should
    # always come out to be [q*, 1, 0, 0, 0, ...]
    #
    
    return qn_star 


def cubic_spline(x,y, boundary = 'natural', boundary_value = (0.0, 0.0),
                 return_jacobian = False):
    """
    Calculate the parameters for a cubic spline.

    Parameters
    ----------
    x : 1d array_like
        The node points.
    y : 1d array_like
        The function vales.
    boundary : {'natural', 'notaknot', 'first', 'second'}, optional
        The boundary condition type. See Notes.
    boundary_value : (2,) tuple, optional 
        The boundary condition values, if applicable. See Notes.
    return_jacobian : bool, optional
        Also return the derivatives of the spline parameters with
        respect to the function values `y`.

    Returns
    -------
    c : (n-1,4) ndarray
        The cubic spline parameters.
    dc : (n,n-1,4) ndarray
        The parameter Jacobian. ``dc[i]`` is the derivative of the
        spline parameters with respect to ``y[i]``. 
        Only returned if `return_jacobian` is ``True``.
    
    Notes
    -----
    The cubic spline consists of a cubic polynomial in each of the 
    :math:`n-1` regions between the :math:`n` node points for a 
    total of :math:`4n-4` parameters. Matching the node values, as well 
    as enforcing continuity of the first and second derivatives at 
    internal nodes, yields only :math:`4n-6` constraints and leaves
    two more to be chosen. The `boundary` parameter determines these 
    two conditions. 
    
    A value of ``'natural'`` sets the second derivatives
    of the spline at the endpoints to 0. 
    
    A value of ``'notaknot'`` uses the not-a-knot condition. The
    third derivative at the first and last interior nodes is
    continuous.
    
    A value of 'first' sets the first derivatives at the boundaries
    equal to the values passed in `boundary_value`.
    
    A value of 'second' sets the second derivatives at the boundaries
    equal to the values passed in `boundary_value`.

    """
    
    x = np.array(x)
    y = np.array(y)
    
    n = len(x) # The number of node points 
    nc = 4*n - 4 # The number of parameters and constraints 
    
    dx = x[1:] - x[:-1] # The step size between neighboring nodes 
    
    # Construct the linear equation matrix
    
    C = np.zeros((nc,nc)) # The coefficient array
    b = np.zeros((nc,))   # The constant vector
    
    db = np.zeros((n,nc)) # The derivative of the constant vector w.r.t y values
    
    #
    # c[4*i + p] is the coefficients of (x-x[i])**p for the 
    # cubic polynomial spanning x[i] to x[i+1] for 
    # i = 0 ... n - 2 
    #

    # Conditions
    # ----------
    # 1) Spline matches value at left boundary
    #
    #     y[i] = c[4*i]
    #
    # 2) Spline matches value at right boundary
    #    
    #     y[i+1] = c[4*i] + c[4*i + 1]*dx[i] + c[4*i + 2]*dx[i]**2 + c[4*i + 3]*dx[i]**3
    #
    idx = 0 # The constraint dummy index 
    for i in range(n-1):
        b[idx] = y[i] 
        db[i,idx] = 1.0 
        
        C[idx,4*i] = 1.0 
        idx += 1 
        
        b[idx] = y[i+1]
        db[i+1,idx] = 1.0 
        
        C[idx,4*i + 0] = 1.0 
        C[idx,4*i + 1] = dx[i]
        C[idx,4*i + 2] = dx[i]**2
        C[idx,4*i + 3] = dx[i]**3  
        idx += 1 
    #
    # 3) Spline derivative is continuous at internal nodes 
    # 
    #  0 = c[4*i + 1] + 2*c[4*i + 2]*dx[i] + 3*c[4*i + 3]*dx[i]**2
    #     - c[4*(i+1) + 1]
    #
    # 
    # 4) Spline second derivative is continuous at internal nodes 
    # 
    #  0 = 2*c[4*i + 2] + 6*c[4*i + 3]*dx[i]
    #     - 2*c[4*(i+1) + 2]
    #
    for i in range(n-2): 
        
        C[idx,4*i + 1] = 1.0 
        C[idx,4*i + 2] = 2*dx[i]
        C[idx,4*i + 3] = 3*dx[i]**2 
        C[idx,4*(i+1) + 1] = -1.0 
        idx += 1 
        
        C[idx,4*i + 2] = 2.0
        C[idx,4*i + 3] = 6*dx[i]
        C[idx,4*(i+1) + 2] = -2.0 
        idx += 1 
        
    # The final two conditions depend on the boundary type
    # 
    if boundary == 'natural':
        #
        # The second derivatives equal 0 at the end points
        #
        C[idx,2] = 2.0 
        idx += 1 
        
        C[idx,-2] = 2.0 
        C[idx,-1] = 6*dx[-1] 
        idx += 1 
    elif boundary == 'notaknot':
        #
        # The third derivatives are continuous at the 
        # first and last interior nodes 
        C[idx,3] = 1.0 
        C[idx,7] = -1.0 
        idx += 1 
        
        C[idx,-5] = 1.0 
        C[idx,-1] = -1.0 
        idx += 1 
    elif boundary == 'first':
        #
        # The first derivative values are passed in 
        # `boundary_value`
        #
        b[idx] = boundary_value[0] 
        C[idx,1] = 1.0 
        idx += 1 
        
        b[idx] = boundary_value[1] 
        C[idx,-3] = 1.0 
        C[idx,-2] = 2 * dx[-1]
        C[idx,-1] = 3 * dx[-1]**2
        idx += 1 
    elif boundary == 'second':
        #
        # The second derivative values are passed in 
        # `boundary_value`
        #
        b[idx] = boundary_value[0] 
        C[idx,2] = 2.0 
        idx += 1 
        
        b[idx] = boundary_value[1] 
        C[idx,-2] = 2.0 
        C[idx,-1] = 6 * dx[-1]**1
        idx += 1 
                
    else:
        raise ValueError("Invalid `boundary` type")
    
    assert (idx == nc)
    # idx should now equal nc !
    #
    # Solve the linear equation b = C @ c 
    # for c
    #
    c = np.linalg.solve(C, b)
    
    # Calculate the derivative of c w.r.t y
    # In general, 
    #    b = C @ c
    # --> db = dC @ c + C @ dc
    #        = C @ dc  ( dC is 0 ... the coefficients do not depend on y)
    # So dc is just another simple linear equation
    dc = np.linalg.solve(C, db.T).T
    
    # Reshape to (n-1, 4)
    #
    c = np.reshape(c, (n-1,4)) 
    dc = np.reshape(dc, (n,n-1,4))
    
    if return_jacobian:
        return c, dc 
    else: 
        return c

def cubic_spline_val(x,c,x0):
    """
    Evaluate a cubic spline 

    Parameters
    ----------
    x : array_like
        The evaluation points.
    c : (n-1,4) ndarray
        The spline parameters.
    x0 : array_like
        The ordered spline nodes.

    Returns
    -------
    y : ndarray
        The spline values at `x`.

    """
    
    n = c.shape[0] + 1 # The original number of nodes 
    
    x = np.array(x) 
    y = np.empty_like(x) 
    
    for i in range(n-1): # For each spline 
        # Determine which `x` values should use
        # this spline
        #
        if i == 0:
            # For the first spline, any point
            # left of the second node 
            #
            mask = x < x0[1] 
        elif i == n - 2:
            # For the final spline, any point 
            # right of the second-to-last node 
            mask = x >= x0[n-2] 
        else:
            # For interior splines
            mask = np.logical_and(x >= x0[i], x < x0[i+1])
        
        y[mask] = c[i,0]\
            + c[i,1] * (x[mask] - x0[i])\
                + c[i,2] * (x[mask] - x0[i])**2\
                    + c[i,3] * (x[mask] - x0[i])**3
    
    return y 

def cubic_spline_derivative(x,c,x0):
    """
    Evaluate the derivative of a cubic spline 

    Parameters
    ----------
    x : array_like
        The evaluation points.
    c : (n-1,5)
        The spline parameters.
    x0 : array_like
        The ordered spline nodes.

    Returns
    -------
    dy : ndarray
        The spline derivatives at `x`.

    """
    
    n = c.shape[0] + 1 # The original number of nodes 
    
    x = np.array(x) 
    y = np.empty_like(x) 
    
    for i in range(n-1): # For each spline 
        # Determine which `x` values should use
        # this spline
        #
        if i == 0:
            # For the first spline, any point
            # left of the second node 
            #
            mask = x < x0[1] 
        elif i == n - 2:
            # For the final spline, any point 
            # right of the second-to-last node 
            mask = x >= x0[n-2] 
        else:
            # For interior splines
            mask = np.logical_and(x >= x0[i], x < x0[i+1])
        
        y[mask] = c[i,1] + 2 * c[i,2] * (x[mask] - x0[i]) + 3 * c[i,3] * (x[mask] - x0[i])**2
    
    return y 

        
class InverseMetric(nitrogen.dfun.DFun):
    """
    An inverse (vibrational) metric function.
    
    
    The derivatives of the lower triangle in packed storage of the 
    inverse metric, or the vibrational block of the inverse metric
    for `bodyframe` embedding, is calculated.

    """
    
    def __init__(self, cs, masses = None, mode = 'bodyframe',
                 planar_axis = None):
        """

        Parameters
        ----------
        cs : CoordSys
            The coordinate system.
        masses : array_like, optional
            The masses. If None, unit masses are assumed.
        mode : {'bodyframe'}, optional
            The embedding mode.
        planar_axis : {None,0,1,2}, optional
            The normal axis for linear/planar coordinate systems. If None,
            this is not used.  See Notes for more details.

        Notes
        -----
        
        The `planar_axis` parameter is used to avoide indeterminances in the 
        inverse metric at linear geometries. In this case, only strictly 
        planar coordinate systems should be used. The block of the metric tensor
        for the two in-plane axes (which is singular)
        decouples from the rest of the metric and 
        can be ignored for calculating the vibrational block of the inverse metric.
        """
        
        nQ = cs.nQ 
        nf = (nQ*(nQ+1)) // 2 # The vibrational block in packed format
        
        maxderiv = (None if cs.maxderiv is None else cs.maxderiv - 1)
        
        super().__init__(self._Gfun, nf = nf, nx = nQ,
                         maxderiv = maxderiv, zlevel = None)
        
        self.cs = cs 
        self.masses = masses 
        self.mode = mode 
        self.planar_axis = planar_axis 
        
        return 
        
        
    def _Gfun(self, Q, deriv = 0, out = None, var = None):
        
        
        # This only support derivatives with all variables
        #
        if var is not None:
            raise NotImplementedError("InverseMetric does not support partial `var` currently.")
        nvar = Q.shape[0]
        
        # Calculate derivatives of the metric tensor, g
        # g is in packed storage
        if self.planar_axis is None: 
            rvar = 'xyz' # Use all axes 
        else:
            rvar = 'xyz'[self.planar_axis] # Remove two in-plane axes 
            
        g = self.cs.Q2g(Q, masses = self.masses, deriv = deriv, mode = self.mode, rvar = rvar)
        
        #
        # Calculate derivatives of the inverse metric, G
        # If planar_axis has been used, then there will be 
        # fewer rotational elements, but the vibrational block
        # of G is unaffected
        #
        G,_ = nitrogen.dfun.sym2invdet(g, deriv, nvar)
        
        if self.mode == 'bodyframe':
            # Return just the vibrational block
            return G[:,:self.nf]
        
        elif self.mode == 'simple':
            # Return all 
            return G 
        
        else:
            raise ValueError(f'Unrecognized mode = {str(self.mode):s}')
            
def symfull_axis(A, axis = 0):
    """
    Expand packed matrix to full symmetric matrix.
    
    Parameters
    ----------
    A : ndarray
        The packed array
    axis : integer, optional
        The packed axis. The default is 0.
        
    Returns
    -------
    Afull : ndarray
        A new array with the packed axis expanded into two symmetric axes.
    """
    
    Aprime = np.moveaxis(A, axis, 0) # Move the packed axis to the front 
    n = Aprime.shape[0] # The packed size 
    N = nitrogen.linalg.packed.n2N(n) # The square size 
   
    Afull = np.empty((N,N) + Aprime.shape[1:], A.dtype)
    
    for i in range(N):
        for j in range(N):
            Afull[i,j] = Aprime[nitrogen.linalg.packed.IJ2k(i,j)]
    
    Afull = np.moveaxis(Afull, (0,1), (axis,axis+1)) # Move the array axes back 
    
    return Afull 


def spline_proxy_path(V, G, q0, q1, proxy_index, nodes,
                      match_level = 0, is_stat = (True,True),
                      max_iter = 20, deltarms = 1e-6):
    """
    Compute a reaction path between two points as a 
    spline function with respect to a proxy coordinate.

    Parameters
    ---------- 
    V : DFun
        The potential energy surface.
    G : DFun
        The inverse metric tensor.
    q0,q1 : (nq,)
        A path end-point.
    proxy_index : integer
        The proxy coordinate index.
    nodes : integer
        The number of interior spline nodes.
    match_level : integer, optional
        The exact boundary condition constraint level. 
        If `match_level` = 0, the path end-points are constrained.
        If `match_level` = 1, the path tangent is also constrained.
        The default is 0.
    is_stat : (2,) tuple of boolean, optional
        Specifies whether each end-point is a stationary point. 
        The default is (True,True). This only matters if `match_level`
        is greater than 0.
    max_iter : integer, optional
        The maximum number of path updates.
    deltarms : float
        The threshold change to the estimated path rms error per step.
        
    Returns
    -------
    path_list : list of (nq,nodes) ndarray
        The path nodes for each update.
    spline_list : list of (nq, nodes-1, 4) ndarray
        The cubic spline parameters for each update.
    rms_list : list of float
        The path rms for each update.
        
    Notes
    -----
    
    An initial guess of the spline nodes is formed by a minimial polynomial 
    interpolantion that meets the matching conditions 
    (a linear interpolant for `match_level` = 0, and a cubic interpolant for
    `match_level` = 1). The positions of the spline nodes are updated by a 
    Newton optimization scheme based on a local quadratic approximation 
    to the path tangent near each node point.
    
    See Also
    --------
    cubic_spline_val : Evaluate the cubic spline functions.
    

    """
    
    ##########################################
    # 
    # Two-point reaction path spline solution
    #
    ##########################################
    
    # Make sure q0 and q1 are in increasing order
    # with respect to the proxy coordinate
    if q0[proxy_index] > q1[proxy_index]:
        q0,q1 = q1,q0 
        is_stat = (is_stat[1], is_stat[0])
    
    ##########################################
    #
    # Calculate the boundary values at the 
    # path end-points. 
    #
    if match_level >= 0: 
        # The path values
        p0,p1 = q0,q1 
        dp0,dp1 = None, None 
    if match_level >= 1:
        # The path tangent is needed
        #
        # First point
        if is_stat[0]: 
            dp0 = qderiv_stationary(q0, 1, V, G)[1]
        else:
            dp0 = qderiv_nonstationary(q0, 1, V, G)[1]
        dp0 = dp0 / dp0[proxy_index]
        
        # Second point 
        if is_stat[1]: 
            dp1 = qderiv_stationary(q1, 1, V, G)[1]
        else:
            dp1 = qderiv_nonstationary(q1, 1, V, G)[1]
        dp1 = dp1 / dp1[proxy_index]
    #
    ##########################################
    
    ########################################## 
    # Construct the initial spline nodes 
    # 
    nq = len(q0)
    path = np.zeros((nq,nodes))
    
    xnodes = np.linspace(p0[proxy_index], p1[proxy_index], nodes)
    
    for i in range(nq):
        if match_level == 0:
            # Linear interpolation of the two end-points 
            path[i] = np.linspace(p0[i], p1[i], nodes)
        elif match_level == 1:
            # Match derivatives at end-points. This requires
            # a cubic polynomial
            x = [p0[proxy_index], p1[proxy_index]] # The matching points 
            df = np.array([ 
                [p0[i],  p1[i]],    # Matching value of coordinate i
                [dp0[i], dp1[i]]    # Matching derivative of coordinate i (w.r.t. proxy)
                ])  
            
            p = nitrogen.math.constrainedPolynomial(x, df)            
            path[i] = np.polyval(np.flip(p), xnodes)
        else:
            raise NotImplementedError("Only match_level = 0 and 1 is supported.")
    #
    # `path` contains the initial trial path
    ##########################################
    
    ##########################################
    #
    # Perform the iterative solution to the spline path 
    old_rms = np.inf 
    old_path = path 
    
    path_list = [old_path] 
    rms_list = [old_rms]
    spline_list = [np.array(_calculate_proxy_spline(path, proxy_index, match_level, dp0, dp1)[0])]
    
    for i in range(max_iter+1):
        
        path,rms = _update_spline_proxy_path(V, G, old_path, proxy_index, 
                                             match_level, dp0, dp1)
        path_list.append(path)
        rms_list.append(rms)
        spline_list.append(
            np.array(_calculate_proxy_spline(path, proxy_index, match_level, dp0, dp1)[0]))
        
        if abs(rms - old_rms) < deltarms:
            break 
        
        old_rms = rms 
        old_path = path 
    
    if i == max_iter:
        print(f"Warning: max iterations {max_iter:d} reached")
        
    
    return path_list, spline_list, rms_list 

def _calculate_proxy_spline(path,proxy_index,match_level,t0,t1):
    
    # Compute the path spline for each coordinate
    # and its derivatives w.r.t. interior nodes
    
    c,dc = [],[] 
    
    x = path[proxy_index] # The x values of the node points
    nq = path.shape[0] 
    
    for i in range(nq): # For each coordinate
        y = path[i]     # The spline function values 
        
        #
        # Calculate the spline parameters.
        #
        if match_level == 0:
            # Value-only end-point conditions
            ci,dci = cubic_spline(x,y,boundary = 'notaknot', return_jacobian = True)
        elif match_level == 1:    
            # Value and tangent end-point conditions
            df = (t0[i], t1[i])
            ci, dci = cubic_spline(x, y, boundary = 'first', 
                                   boundary_value = df, return_jacobian = True)
        else:
            raise NotImplementedError("Only match_level = 0 or 1 is supported.")
        
        c.append(ci)
        #
        # We only need the derivatives of the spline parameters with respect to the
        # **interior** node values. The first and last are fixed.
        #
        dc.append(dci[1:-1])
    return c,dc 
        
def _update_spline_proxy_path(V, G, path, proxy_index,
                              match_level, t0, t1):
    """
    Perform a single spline-proxy path update 

    Parameters
    ----------
    V : DFun
        The potential energy surface.
    G : DFun
        The inverse metric tensor.
    path : (nq,nodes) ndarray
        The current path nodes.
    proxy_index : integer
        The proxy coordinate index 
    match_level : integer
        The matching level, 0 or 1.
    t0,t1: (nq,) ndarray
        The end-point path tangent (for matching_level == 1).

    Returns
    -------
    new_path : (nq,nodes) ndarray
        The updated spline nodes.
    rms : float
        The estimated root-mean-square path error in mass-weighted distance.

    """
    #
    # Given the current path nodes : 
    #
    # 1) Evaluate the path and its derivatives w.r.t. node values
    #    at the segment mid-points
    #
    # 2) Evaluate the exact path tangent and derivative of the tangent
    #    w.r.t coordinates at the interpolated path position at the
    #    segment mid-points
    #
    # 3) Solve the least-squares system for matching the path tangent
    #
    
    # -----------------------------
    # 
    # Calculate the spline and its parameter derivatives for the 
    # current path.
    #
    nq = path.shape[0]     # The number of coordinates 
    nnode = path.shape[1]  # The total number of nodes, including end-points 
    nsample = nnode - 1    # The number of sample points (i.e. segments)
    
    #
    # The number of **interior** nodes is nnode - 2 
    #
    
    c,dc = _calculate_proxy_spline(path,proxy_index,match_level,t0,t1)
        
    # Evaluate the spline at the mid-point of each segment
    x = path[proxy_index] # The x values of the node points
    x_sample = (x[1:] + x[:-1]) / 2.0 
    p_sample = np.array([cubic_spline_val(x_sample, ci, x) for ci in c])
    # p_sample[i,j] is the value of the i**th coordinate at the j**th sample point
    dp_sample = np.array([
            [cubic_spline_val(x_sample, dci[k], x) for dci in dc]
            for k in range(len(x)-2)])
    #
    # dp_sample[k,i,j] is the derivative of the i**th coordinate value at the j**th
    # sample point with respect to the k**th interior node value of the i**th coordinate
    #
    
    # Evaluate the spline tangent at the mid-point of each segment
    t_sample = np.array([cubic_spline_derivative(x_sample, ci, x) for ci in c])
    #
    # and the deriative of the spline tangent at the sample points with respect to the
    # node values
    #
    dt_sample = np.array([
        [cubic_spline_derivative(x_sample, dci[k], x) for dci in dc]
        for k in range(len(x)-2)])
    #
    # Calculate the true path tangent and derivative thereof 
    # at each sample point 
    #
    w0,W,G0 = LQA_nonstationary(p_sample, V, G, arclength = 'proxy', proxy_index = proxy_index)
    
    def matrix_inversesqrt(A):
        #
        # A ... (n,n,...) real symmetric
        #
        A = np.moveaxis(A, (0,1), (-2,-1)) # move array indices to end 
        w,U = np.linalg.eigh(A)
        # A = U @ diag[w] @ U.T
        # irt[A] = U @ diag[1/sqrt[w]] @ U.T 
        irtA = np.einsum('...ij,...j,...kj->...ik', U, 1.0 / np.sqrt(w), U) 
        # irtA currently has shape (...,n,n)
        irtA = np.moveaxis(irtA, (-2,-1), (0,1)) # move back to front 
        return irtA 
    irtG = matrix_inversesqrt(G0)
    
    kron = np.eye(nq) # kronecker delta
    
    # 
    # Construct the least-squares array and vector
    #
    C = np.einsum('ilj,klj->ijlk',W,dp_sample) - np.einsum('il,klj->ijlk',kron,dt_sample)
    b = t_sample - w0 
    
    # Apply weighting by G^-1/2 for "proper" mass-weighted least squares
    C = np.einsum('imj,mjlk->ijlk', irtG, C) 
    b = np.einsum('imj,mj->ij', irtG, b) 
    
    # Reshape to matrix-vector problem
    C = np.reshape(C, (nq*nsample, nq*(nsample-1))) 
    b = np.reshape(b, (nq*nsample,))
    
    #
    # Solve C @ dy = b by least-squares
    #
    dy,res,_,_ = np.linalg.lstsq(C,b, rcond = None) 
    dy = np.reshape(dy, (nq,nsample-1))
    #
    # dy[i,k] is the correction to the i**th coordinate at the k**th interior node value 
    
    # `res` is the sum of squared residuals.
    # Because the residuals are already mass-weighted,
    # the rms of `res` gives the average path error in `proper` mass-weighted distance.
    #
    rms = np.sqrt(res / nsample)
    
    new_path = path.copy() 
    new_path[:,1:-1] += dy 
    
    return new_path , rms 

def christoffel_symbol(q, G, kind = 'first'):
    """
    Calculate the Christoffel symbols, :math:`\\Gamma_{ijk}` or :math:`\\Gamma^i{}_{jk}`.

    Parameters
    ----------
    q : (nq,...) array_like
        The evaluation points.
    G : DFun
        The inverse metric tensor, :math:`G^{ij}`.
    kind : {'first','second'}, optional
        Calculate symbols of the first or second kind. 'first' is the default.
    
    Returns
    -------
    Gamma : (nq,nq,nq,...) ndarray
        The Christoffel symbols.
        
    Notes
    -----
    
    The Christoffel symbols of the first kind are
    
    ..  math::
        
        \\Gamma_{ijk} = \\frac{1}{2}\\left( \\partial_k g_{ij} + \\partial_j g_{ik} - \\partial_i g_{jk} \\right)


    The Christoffel symbols of the second kind are
    
    ..  math::
        
        \\Gamma^i{}_{jk} = G^{im} \\Gamma_{mjk}
        
    Either kind is symmetric in the last two indices.
    
    """
    
    # Calculate the inverse metric G
    # and its first derivatives 
    
    q = np.array(q)
    nq = q.shape[0] 
    base_shape = q.shape[1:]
    
    if kind != 'first' and kind != 'second':
        raise ValueError("kind must be 'first' or 'second'")
    
    dG = symfull_axis(G.f(q, deriv = 1), axis = 1)
    G0 = dG[0]   # G[i,j]
    G1 = dG[1:]  # d_i G[j,k]
    
    # Calculate the metric tensor, g, 
    # and its inverse 
    g0 = np.linalg.inv(np.moveaxis(G0,(0,1), (-2,-1)))
    g0 = np.moveaxis(g0, (-2,-1), (0,1))
    #
    #      G @ g = I
    # -->  (dG) @ g + G @ (dg) = 0 
    # -->   dg = -g @ dG @ g
    #
    g1 = -np.einsum('jl...,ilm...,mk...->ijk...', g0, G1, g0)
    
    # g0 ... g[i,j]
    # g1 ... d_i g[j,k]
    #
    
    # Calculate Christoffel symbols of the first kind 
    Gamma = np.empty((nq,nq,nq) + base_shape, dtype = g0.dtype)
    
    for c in range(nq):
        for a in range(nq):
            for b in range(nq):
                
                val = 0.5 * (g1[b,c,a] + g1[a,c,b] - g1[c,a,b])
                
                if len(base_shape) == 0: # Single-point
                    Gamma[c,a,b] = val   # Assign scalar value
                else:
                    np.copyto(Gamma[c,a,b], val) 
                    
    if kind == 'first':
        return Gamma 
    else:
        # Calculate symbols of the second kind 
        Gamma = np.einsum('il...,ljk...->ijk...',G0,Gamma)
        return Gamma 
    
def covariant_hessian(q,V,G):
    """
    Calculate the covariant Hessian tensor, :math:`\\nabla_i \\nabla_j V`.

    Parameters
    ----------
    q : (nq,...)
        The evaluation points.
    V : DFun
        The potential energy surface.
    G : DFun
        The inverse metric tensor.

    Returns
    -------
    f : (nq,...)
        The gradient.
    H : (nq,nq,...)
        The covariant Hessian.
        
    Notes
    -----
    
    The covariant gradient is equal to the regular gradient, 
    :math:`\\nabla_i V = \\partial_i V`. The covariant Hessian is
    
    ..  math::
        
        H_{ij} &= \\nabla_i \\nabla_j V \\
            
               &= \\nabla_i f_j \\ 
            
               &= \\partial_i \\partial_j V - f_k \\Gamma^k{}_{ij}

    The covariant Hessian is symmetric, :math:`H_{ij} = H_{ji}`.
    
    """
    
    
    # Calculate the regular gradient and Hessian.
    _,f,K = [a[0] for a in V.vjh(q)]
    
    # Calculate the Christoffel symbols
    # (of the second kind)
    Gamma = christoffel_symbol(q, G, kind = 'second')
    
    # Calculate the covariant Hessian
    #
    # K_ij = d_i f_j  ...  the regular Hessian
    # 
    # H_ij = D_i f_j  ...  the covariant Hessian (where D_i is covariant deriv.)
    #
    #      = d_i f_j - f_k Gamma^k_ij
    #
    
    H = K - np.einsum('k...,kij...->ij...', f, Gamma) 
    
    return f,H

def pathvib_nonstationary(q, V, G, hbar = None):
    """
    Calculate reaction path normal modes
    using the orthogonally projected covariant Hessian.

    Parameters
    ----------
    q : (nq,...) array_like
        The evaluation points at non-stationary geometries.
    V : DFun
        The potential energy surface.
    G : DFun
        The inverse metric tensor.
    hbar : float, optional
        The value of :math:`\\hbar`. If None, the default
        NITROGEN units will be used. 

    Returns
    -------
    omega : (nq-1,...) ndarray
        The orthogonal path frequencies in energy units.
    T : (nq,nq-1,...) ndarray
        The displacement vectors of the `nq`-1 orthogonal modes, 
        normalized as reduced dimensionless normal coordinates.
        ``T[i,j]`` is the displacement of coordinate ``i`` for 
        unit amplitude of normal mode ``j``.
    Tl : (nq,nq-1,...) ndarray
        The lowered-index transformation of `T`.
    """
    
    ####################################
    #
    # The covariant Hessian will be evaluated
    # at each point and projected against the
    # local gradient. The remaining non-zero-
    # frequency vibrations will be calculated.
    #
    q = np.array(q)
    nq = q.shape[0] 
    base_shape = q.shape[1:]
    
    
    f,H = covariant_hessian(q, V, G)
    G0 = symfull_axis(G.f(q, deriv = 0)[0], axis = 0)
    # We will also need the inverse of G later 
    iG = np.linalg.inv(np.moveaxis(G0, (0,1), (-2,-1)))
    iG = np.moveaxis(iG, (-2,-1), (0,1)) 
    
    # Raise the first index of H_{ij}
    Hp = np.einsum('ij...,jk...->ik...',G0,H)
    
    # Calculate the projection matrix P
    # 
    # P^i_j = delta_ij - f^i f_j / |f|^2
    #
    # where f is the local gradient
    #
    # The normal gradient `f` is f_j with a lowered index
    # Calculate F = f^i, with an upper index
    F = np.einsum('ij...,j...->i...', G0, f) 
    f2 = np.sum(F*f,axis=0) 
    
    # Construct kronecker delta
    delta = np.zeros((nq,nq) + base_shape) 
    for i in range(nq):
        delta[i,i] = 1.0  
    
    P = delta - np.einsum('i...,j...->ij...',F,f)/f2 
    
    #
    # Project the Hessian ... P.H.P
    #
    Htilde = np.einsum('ij...,jk...,kl...->il...',P,Hp,P)
    
    #
    # Calculate its eigenvalues and eigenvectors 
    #
    M = np.moveaxis(Htilde,(0,1), (-2,-1)) # move axes to right
    w,U = np.linalg.eig(M)
    w = np.moveaxis(w, -1, 0)
    U = np.moveaxis(U, (-2,-1), (0,1)) 
    
    # w contains the eigenvalues of the projected Hessian
    # One eigenvalue is zero, corresponding to the projected-out gradient vector
    #
    # Sort the eigenvalues by magnitude
    # (Note use of take_along_axis for arbitrary base_shape)
    I = np.argsort(abs(w), axis = 0)
    w = np.take_along_axis(w, I, axis = 0)
    U = np.array([np.take_along_axis(U[i], I, axis = 0) for i in range(nq)])
    
    # Check that the zero-frequency eigenvector is parallel to gradient
    # (i.e. that there is not some accidental zero-frequeny vector from an
    #  orthogonal mode)
    U0 = U[:,0] * np.sqrt(np.sum(F*F,axis=0))
    cos = abs( np.sum(f*U0,axis=0) / f2 )
    if np.any( (1.0 - cos) > 1e-5 ):
        print("Warning: Zero-frequency vector may be mis-identified.")
    
    #
    # Normalize orthogonal vibration displacements
    # to the usual reduced dimensionless coordinates
    # 
    # i.e. V = 0.5 * omega * q^2
    #
    if hbar is None:
        hbar = nitrogen.constants.hbar 
    
    w_ortho = w[1:] # The orthogonal eigenvalues
    
    # Calculate orthogonal harmonic frequencies
    omega = hbar * np.sqrt(np.abs(w_ortho))  # Calculate harmonic energies
    omega[w_ortho < 0] = -omega[w_ortho < 0] # Imaginary frequencies will be flagged as negative
     
    # Calculate the normal coordinate transformation matrix
    # for the "dimensionless normal coordinates" which are
    # normalized as V = 0.5 * omega * q^2
    #
    Uortho = U[:,1:] # The orthogonal displacement vectors 
    
    T = Uortho.copy() 
    for i in range(nq-1):
        ui = Uortho[:,i] 
        # Hii = Ui.T @ Htilde @ ui
        #     = ui @ G^-1  @ Htilde @ ui 
        # (Htilde is calculated as Htilde ^i _j)
        #
        Hii = np.einsum('k...,ki...,ij...,j...', ui, iG, Htilde, ui)
        
        T[:,i] *= np.sqrt( np.abs(omega[i] / Hii) )

    Tl = np.einsum('ij...,jk...->ik...',iG,T)
    
    return omega, T, Tl 

def correct_vib_order(omega, T, Tl, hbar = None):
    """
    Correct the ordering and phase of reaction path vibrational modes.

    Parameters
    ----------
    omega : (nq,N) ndarray
        The harmonic frequncies (in energy units.)
    T : (nq,nm,N) ndarray 
        The displacement vectors normalized as reduced dimensionless
        normal modes.
    Tl : (nq,nm,N) ndarray 
        The left-hand displacement vectors (i.e. `T` multiplied by the 
        effective metric tensor).
    hbar : float, optional
        The value of :math:`\\hbar`. If None, the default
        NITROGEN units will be used. 

    Returns
    -------
    omega_new : (nq,N) ndarray
        The ordered frequencies
    T_new : (nq,nm,N) ndarray 
        The ordered displacement vectors.
    Tl_new : (nq,nm,N) ndarray 
        The ordered left-hand vectors.
    
    See Also
    --------
    pathvib_nonstationary : Calculates `omega`, `T`, and `Tl`.

    """
    
    ##########################################
    #
    # We assume the vibrational coordinates have been calculated
    # along a sufficiently smooth 1D path (e.g. a reaction path)
    # 
    # We will compare the vectors point-by-point to correct
    # the mode ordering and phase (i.e. sign)
    #
    # Because of different coordinate scaling, the only consistent
    # comparison between displacement vectors is via the
    # proper inner-product between right- and left-hand vectors
    #
    # 
    
    omega_new = omega.copy() 
    T_new = T.copy() 
    Tl_new = Tl.copy() 
    
    if hbar is None:
        hbar = nitrogen.constants.hbar 
        
    nq,nm,N = T.shape 
    # nq ... the number of coordinates
    # nm ... the number of modes (typically nq - 1)
    # N  ... the number of points in the 1D path
    
    # We assume the first point is already correct.
    # This defines the reference phase and ordering 
    
    for i in range(1,N): 
        # Move sequentially through points 
        
        T_prev = T_new[:,:,i-1]  # The previous point's right-hand vectors
        Tl_curr = Tl[:,:,i]  # The current left-hand vectors 

        idx_best = [None] * nm 
        
        for j in range(nm): # For the j**th mode
            # Find the vector which is closest 
            max_product = 0.0 
            max_k = -1
            
            Tj = T_prev[:,j]
            omegaj = omega_new[j,i-1]
            
            for k in range(nm):
                tk = Tl_curr[:,k]
                omegak = omega_new[k,i]
                
                # Compute inner product of T_j and t_k
                #
                # The reduced dimensionless normal coordinates
                # are normalized such that the dot product 
                # of the left and right vectors equal
                # hbar**2 / omega 
                # (where omega is the harmonic energy)
                #
                
                prod = np.sum(Tj*tk) * np.sqrt(abs(omegaj*omegak)) / hbar**2
                prod = abs(prod)
                
                if prod > max_product:
                    max_product = prod
                    max_k = k 
                
                idx_best[j] = max_k
        
        if len(np.unique(idx_best)) != len(idx_best):
            print(f"Warning: duplicate match! (point i = {i:d})")
            print(idx_best)
        
        omega_new[:,i] = omega[idx_best,i]
        for j in range(nm):
            tnew = T[:,idx_best[j],i].copy()
            if np.sum(tnew * T_prev[:,j]) < 0.0:
                tnew *= -1.0 
            np.copyto(T_new[:,j,i], tnew) 
            
    return omega_new, T_new, Tl_new 
        
    