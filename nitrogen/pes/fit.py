"""
fit.py

Surface fitting tools
"""

import numpy as np 
import nitrogen 
import nitrogen.autodiff.forward as adf 

def fitSimplePIP(X,F,P,yfun,degree,Xscale):
    """
    Fit a permutationally invariant polynomial 
    in terms of an internuclear distance function.

    Parameters
    ----------
    X : (3*n,N) ndarray
        The Cartesian coordinates of `n` particles at `N` sampling points.
    F : (nd,N) ndarray
        The derivative array of the function values. If
        only the value is fitted, for example, `nd` is 1.
    P : list
        The permutation elements. Each element of `P` is a
        permutation of ``[0,1,2,...,n-1]``. 
    yfun : DFun
        The internuclear coordinate function. It takes
        as its argument the (3*n,...) Cartesian coordinates
        and returns the ``n*(n-1)//2`` internuclear functions.
        These functions need to be consisten with the permutational
        symmetries implied by `P`.
    degree : integer
        The polynomial degree.
    Xscale : float
        The Cartesian length scale. Its powers will be used to scale
        derivatives to the value units.

    Returns
    -------
    p : ndarray
        The expansion coefficients
    res : ndarray
        The scaled residuals.
    Ffun : DFun
        The fitted function, F(X).
    
    Notes
    -----
    Derivatives up to arbitrary order can be fitted.
    
    See Also
    --------
    InternuclearExp : internuclear function for exponentially scaled distance
    InternuclearR : internuclear function for linear distance 

    """
    
    
    #####################################
    # A simple PIP fitting routine
    # for small molecular permutation groups
    #
    natoms = X.shape[0] // 3 # the number of atoms 
    ny = (natoms*(natoms-1)) // 2 # The number of internuclear y coordinates
    
    #
    # The `y` variables are the n*(n-1)//2 functions
    # of the internuclear distances. This can be any
    # 'radial' function, e.g. r12, exp(-r12/a), etc.,
    # which is passed by the caller.
    #
    
    # Calculate the y-index permutations from the 
    # atomic permutations
    Py = atom2yperm(P)
    
    # Create a list of y-powers for each term 
    pows = adf.idxtab(degree, ny) 
    nt = pows.shape[0] # The number of monomial terms 
    
    # Define a function which returns the lexical maximum
    # of two lists of powers 
    def lexical_max(idx1, idx2):
        # Find the lexical maximum of two indices
        #
        # Rules
        # -----
        # 1) First, sort by sum (i.e. total degree)
        # 2) If the degree is equal, then sort by first index
        # 3) If the first index is equal, then sort by second, etc.
        #
        if len(idx1) != len(idx2):
            raise ValueError("Indices must be of equal length")
        
        if sum(idx1) > sum(idx2):
            return idx1 
        elif sum(idx1) < sum(idx2):
            return idx2 
        
        # degrees are equal
        for i in range(len(idx1)):
            if idx1[i] > idx2[i]:
                return idx1 
            elif idx1[i] < idx2[i]:
                return idx2 
            # else, continue
        
        # If we exit the for the loop, then
        # the indices are equal. Just 
        # return idx1 
        return idx1 
    
    # We now figure out the symmetric projection of the 
    # list of monomials
    # 
    # For each monomial, find the lexical maximum of its invariant projection
    # This is the "representative monomial" or "representative index"
    #
    
    nck = adf.ncktab(ny+degree-1,min(ny,degree-1)) # A binomial coefficient table
    
    # unique_term_pos will keep track of which 
    # representative monomial each monomial projects to 
    #
    unique_term_pos = np.zeros((nt,), dtype = np.uint32)
    for i in range(nt):
        # For each monomial
        pi = pows[i,:] # The monomial's power index
        
        max_idx = pi 
        # Find the lexical maximum of
        # all of its permutations 
        for py in Py: 
            perm_pi = pi[py] # The permuted monomial
            max_idx = lexical_max(max_idx, perm_pi) 
        
        # max_idx serves as a unique label for
        # the invariant sum
        unique_term_pos[i] = adf.idxpos(max_idx, nck)

    # Get the sorted list of unique, representative monomials
    rep_terms = np.sort(np.unique(unique_term_pos))
    nr = len(rep_terms) # unique terms after projection 
    
    print(f"There are {ny:d} pair-coordinates.")
    print(f"For degree = {degree:d}, there are {nt:d} monomials.")
    print(f"After projection, there are {nr:d} invariant terms.")
    print("")
    print("Calculating least-squares matrix...", end = "")
    
    # `rep_terms` will serve as the order of the columns of the 
    # least-squares array. Figure out which column each
    # monomial maps to
    monomial_map = np.searchsorted(rep_terms, unique_term_pos)
    
    # Create the least-squares array
    # Initialize to zero 
    
    data_nd = F.shape[0] # The number of Cartesian derivatives supplied by caller
    npoints = F.shape[1] # The number of geometries 
    data_deg = nitrogen.dfun.infer_deriv(data_nd, 3*natoms) # The derivative order
    # of the data
    
    C = np.zeros((data_nd,npoints,nr))
    
    
    # Calculate the value/derivatives of each monomial term of the 
    # surface expansion w.r.t. X coordinates
    Z_of_y = nitrogen.dfun.PowerExpansionTerms(degree, np.zeros((ny,)))
    Z_of_X = yfun ** Z_of_y 

    ymonomials = Z_of_X.f(X, deriv = data_deg)  # shape (data_nd, nt, npoints)
    
    # Accumulate each monomial into the correct
    # final fitting term
    for r in range(nt):
        # Add this monomial to the appropriate column of C
        C[:,:,monomial_map[r]] += ymonomials[:,r,:]
    
    # Multiply by approriate length scale
    # passed as Xscale 
    #
    Fcopy = np.copy(F) # A copy of F, which we can modify 
    
    for deg in range(1,data_deg+1):
        start = nitrogen.dfun.nderiv(deg-1, 3*natoms)
        stop = nitrogen.dfun.nderiv(deg, 3*natoms)
        
        C[start:stop] *= Xscale**deg 
        Fcopy[start:stop] *= Xscale**deg

    # Now reshape C 
    C = C.reshape((data_nd*npoints, nr))
    # and the fit data 
    b = Fcopy.reshape((data_nd*npoints,))
    print("done")
    
    print("Calculating least-squares solution...", end = "")
    # Now solve the linear least-squares problem
    p,_,_,_ = np.linalg.lstsq(C, b, rcond=None)
    res = b - C @ p
    #        
    print("done")
    print("")
    #
    # Convert the fitted parameters to the 
    # full list of individual monomials
    pfull = np.zeros((nt,))
    for r in range(nt):
        pfull[r] = p[monomial_map[r]]
    
    
    ########################
    # Results        
    rmse = np.sqrt(np.average(res**2)) # Total scaled RMS residual
    print("--------------------------------------")
    print(f"Total scaled rmse = {rmse:.3f}")
    print("--------------------------------------")
    # Partition RMS error by derivative order 
    res = res.reshape((data_nd,npoints))
    
    start = 0
    for deg in range(data_deg+1):
        stop = nitrogen.dfun.nderiv(deg, 3*natoms)
        rmse_deg = np.sqrt(np.average(res[start:stop]**2))
        print(f"Degree {deg:d} rmse     = {rmse_deg:.3f}")
        start = stop
    print("--------------------------------------")
    print("")
    
    #
    # Create the DFun object for F(X) function
    #
    surface_fun = yfun ** nitrogen.dfun.PowerExpansion(pfull.reshape((-1,1)),
                                                       np.zeros((ny,)))
    
    return pfull, res, surface_fun
    

def atom2yidx(n):
    """
    Calculate the 2d table mapping pairs of 
    atomic indices to y-variable indices.

    Parameters
    ----------
    n : int
        The atom count.
        
    Returns
    -------
    yidx : ndarray
        The map of atom indices to y indices.
        The map is symmetric, y[i,j] = y[j,i],
        and only the off-diagonal elements are valid.

    """
    # Create a table mapping atom pairs [i,j]
    # to y variables
    yidx = np.zeros((n,n), dtype = np.int32)
    idx = 0 
    for i in range(n):
        for j in range(i+1,n):
            yidx[i,j] = idx 
            yidx[j,i] = idx 
            idx += 1 
    
    return yidx 
    
def atom2yperm(P):
    """
    Convert a list of permutations P of atoms
    to a list of permutation of y variables.

    Parameters
    ----------
    P : list
        The atomic permutations.

    Returns
    -------
    Py : list
        The y-variable permutations

    """
    
    #
    if not P: # List is empty
        return [] # Just return another empty list 
    
    # Otherwise, continue
    
    # The atom count equals the length of the
    # elements of P
    #
    n = len(P[0])
    
    
    yidx = atom2yidx(n) # Get a map of atom pairs to y variables
    
    Py = [] 
    for p in P: # For each permutation element
    
        # Construct the y-variable permutation
        py = []
        for i in range(n):
            for j in range(i+1,n):
                # Variable y_ij maps to y_{p[i], p[j]}
                
                new_idx = yidx[p[i], p[j]]
                py.append(new_idx)
        
        Py.append(py)
    
    return Py 

class InternuclearExp(nitrogen.dfun.DFun):
    """
    Internuclear coordinate function
    for exponential (Morse) scaled distance
    
    Attributes
    ----------
    a : float
        The exponential length parameter
    n : integer
        The number of atoms.
    """

    def __init__(self, n, a):
        """

        Parameters
        ----------
        n : integer
            The number of atoms
        a : float
            The exponential length parameter.

        """
        
        if n < 2 :
            raise ValueError("There must be at least 2 atoms.")
        
        if a <= 0:
            raise ValueError("`a` must be positive")
        
        # The number of inputs is 3*n, the number of 
        # Cartesian coordinates 
        nx = 3*n 
        
        # The number of outputs is the number of 
        # atoms pairs
        nf = (n*(n-1)) // 2 
        
        super().__init__(self._fexpy, nf = nf, nx = nx,
                         maxderiv = None, zlevel = None)
        
        self.n = n
        self.a = a 
        
        return 
    
    
    def _fexpy(self, X, deriv = 0, out = None, var = None):
        """ evaluation function """
        
        x = nitrogen.dfun.X2adf(X, deriv, var)
        
        n = self.n 
        
        y = [] 
        for i in range(n):
            for j in range(i+1,n):
                
                dx = x[3*i + 0] - x[3*j + 0]
                dy = x[3*i + 1] - x[3*j + 1]
                dz = x[3*i + 2] - x[3*j + 2]
                
                rij = adf.sqrt( dx*dx + dy*dy + dz*dz ) 
                yij = adf.exp(-rij / self.a) 
                y.append(yij)
                
        return nitrogen.dfun.adf2array(y, out)

class InternuclearR(nitrogen.dfun.DFun):
    """
    Internuclear coordinate function
    for linear distance
    
    Attributes
    ----------
    n : integer
        The number of atoms.
    """

    def __init__(self, n):
        """

        Parameters
        ----------
        n : integer
            The number of atoms

        """
        
        if n < 2 :
            raise ValueError("There must be at least 2 atoms.")
            
        # The number of inputs is 3*n, the number of 
        # Cartesian coordinates 
        nx = 3*n 
        
        # The number of outputs is the number of 
        # atoms pairs
        nf = (n*(n-1)) // 2 
        
        super().__init__(self._fdist, nf = nf, nx = nx,
                         maxderiv = None, zlevel = None)
        
        self.n = n
        
        return 
    
    
    def _fdist(self, X, deriv = 0, out = None, var = None):
        """ evaluation function """
        
        x = nitrogen.dfun.X2adf(X, deriv, var)
        
        n = self.n 
        
        y = [] 
        for i in range(n):
            for j in range(i+1,n):
                
                dx = x[3*i + 0] - x[3*j + 0]
                dy = x[3*i + 1] - x[3*j + 1]
                dz = x[3*i + 2] - x[3*j + 2]
                
                rij = adf.sqrt( dx*dx + dy*dy + dz*dz ) 
                y.append(rij)
                
        return nitrogen.dfun.adf2array(y, out)
    
def Sn(n,indices):
    """
    Return the permutations of identical
    particles.
    
    Parameters
    ----------
    n : integer 
        The total number of particles
    indices : array_like
        A list of identical indices (`0` through `n`-1).
    
    Returns
    -------
    P : list
        A list of permutations
    """
    
    # First, a simple recursive generator
    # of all permutations of a list 
    def perm(start, end=[]):
        if(len(start) == 0):
            return [end]
        else:
            perms = []
            for i in range(len(start)):
                perms = perms + perm(start[:i] + start[i+1:], end + start[i:i+1])
            return perms 
    
    S = perm(indices) 
    
    P = [] 
    for s in S:
        p = [i for i in range(n)]
        for i in range(len(indices)):
            p[indices[i]] = s[i]
        P.append(p)
       
    return P 
    
def productP(P1,P2):
    """
    Return the direct product of all
    permutations in P1 and P2.
    
    Parameters
    ----------
    P1,P2 : list
        A list of permutations
    
    Returns
    -------
    P : list
        The direct products
    
    """
    
    n1 = len(P1)
    n2 = len(P2)
    
    P = [] 
    
    for i in range(n1):
        p1 = P1[i] 
        
        for j in range(n2):
            p2 = P2[j] 
            
            p = [p2[idx] for idx in p1]
            
            P.append(p)
    
    return P 
    
def fitFourier(x,y,max_freq,period=None,symmetry=None):
    """
    Fit a Fourier series.

    Parameters
    ----------
    x : (N,) array_like
        The argument at `N` sampling points.
    y : (N,) or (m,N) array_like
        The values of one or more (`m`) functions.
    max_freq : integer
        The maximum frequency harmonic.
    period : float, optional.
        The period of `x`. If None (default), `period` = :math:`2\\pi` is assumed.
    symmetry : integer or (m,) array_like, optional
        The symmetry type of each function. If None (default), no symmetry
        is assumed. See Notes.

    Returns
    -------
    (n,) or (m,n) ndarray
        The Fourier coefficients

    Notes
    -----
    
    The expansion coefficients are defined as 
    
    ..  math::
        
        f(x) = c_0 + c_1 \\sin \\sigma x + c_2 \\cos \\sigma  x +
        c_3 \\sin 2 \\sigma  x + c_4 \\cos 2 \\sigma  x + \\cdots
        
    where :math:`\\sigma = 2\\pi/`\ `period`.
    
    The `symmetry` keyword specifies a constraint on the Fourier series for
    each fitted function. A value of ``0`` fits all terms, ``1``
    fits only cosine terms, ``2`` fits only sine terms, and ``-1`` fixes all
    parameters to 0.
    
    See Also
    --------
    :func:`nitrogen.dfun.FourierSeries` : A Fourier series DFun.
    
    """
    
    if period is None:
        scale = 1.0 
    else:
        scale = 2*np.pi / period 
        
    x = np.array(x)
    y = np.array(y)
    is_scalar = False 
    
    if y.ndim == 1:
        y = np.reshape((1,-1))
        is_scalar = True 
    m,N = y.shape # the number of functions to fit and the number of points 
    
    n = 1 + 2*max_freq # The number of coefficients 
    
    if symmetry is None:
        symmetry = [0] * m 
    elif np.isscalar(symmetry):
        symmetry = [symmetry]
    
    # Construct the least-squares matrix for no symmetry 
    C = np.ones((N,n))
    
    for i in range(1,max_freq+1):
        C[:,2*i-1] = np.sin(x * i * scale)
        C[:,2*i]   = np.cos(x * i * scale) 
    
    ########################
    # Fit each function 
    
    c = np.zeros((m,n)) # the Fourier coefficients 
    
    for k in range(m):
        
        #
        # Function k
        # c[k] has been initialized to 0's
        #
        if symmetry[k] == -1:
            continue  # All zeros
        elif symmetry[k] == 0: 
            # Fit all terms
            ck,_,_,_ = np.linalg.lstsq(C, y[k], rcond=None)
            c[k] = ck
        elif symmetry[k] == 1: 
            # Fit only cosine (even terms)
            ck,_,_,_ = np.linalg.lstsq(C[:,::2], y[k], rcond=None)
            c[k,::2] = ck
        elif symmetry[k] == 2: 
            # Fit only sine (odd terms)
            ck,_,_,_ = np.linalg.lstsq(C[:,1::2], y[k], rcond=None)
            c[k,1::2] = ck
        else:
            raise ValueError("Invalid `symmetry` value")
    
    # All fits are complete 
    
    if is_scalar:
        # Reshape c to 1d
        c = c.reshape((n,))
        
    return c 
        
        
    