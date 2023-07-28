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
            # (Note, I think pi[py] is actual the inverse permutation
            #  w.r.t to how I have defined the y permutation operators.
            #  However, the total set of images is the same either way,
            #  so the final lexical maximum is unchanged!)
            #
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
                ##################################
                # Be careful of the P operator convention.
                #
                # For a given P, P[i] is the new label of 
                # the original atom i 
                # 
                # The atomic P acts on the Cartesian coordinates
                # of atom i as 
                #
                # P[ X_i ] = X_I  such that  P[I] = i  (**not P[i] = I**)
                #
                # Therefore 
                #
                # P[ y_ij ] = y_IJ  where P[I] = i and P[J] = j 
                #
                I = p.index(i)
                J = p.index(j) 
                
                new_idx = yidx[I,J]
                py.append(new_idx)
        
        Py.append(py)
    
    return Py 

def atom2vperm(P,Vij):
    """
    Convert a list of permutations P of atoms
    to a list of permutation of bond-pair vectors,
    including the sign of the permutation.

    Parameters
    ----------
    P : list
        The atomic permutations.
    Vij : list of (2,)
        The bond pairs.
    
    Returns
    -------
    Pv : list
        The bond vector permutations
    Sv : list
        The sign of the permutations.

    """
    
    if not P: # List is empty
        return [], [] # Just return empty lists

    nv = len(Vij) # The number of bond-vectors 
    
    Pv = [] 
    Sv = [] 
    
    for p in P: # For each permutation element
    
        # Construct the bond vector permutation
        pv = [] 
        sv = [] 
        for k in range(nv):
            # For the k**th bond vector
            i,j = Vij[k] # The atom labels 
            
            ##################################
            # Be careful of the P operator convention.
            #
            # For a given P, P[i] is the new label of 
            # the original atom i 
            # 
            # The atomic P acts on the Cartesian coordinates
            # of atom i as 
            #
            # P[ X_i ] = X_I  such that  P[I] = i  (**not P[i] = I**)
            #
            #
            I = p.index(i)
            J = p.index(j) 
            
            # P[ V(i-->j) ] = P[ Xj - Xi ]
            #               = XJ - XI 
            #               = V(I-->J)
            #
            # Find the bond vector (I,J)
            # or (J,I). If (I,J) exists, then
            # the sign is +1. If (J,I) then the sign is -1
            #
            found = False 
            for K in range(nv):
                if Vij[K][0] == I and Vij[K][1] == J:
                    # Vector K is the result, with positive sign
                    pv.append(K)
                    sv.append(+1)
                    found = True 
                    break 
                elif Vij[K][0] == J and Vij[K][1] == I:
                    # Vector K is the result, but with oppositive sign 
                    pv.append(K)
                    sv.append(-1)
                    found = True 
                    break 
            
            if not found:
                raise ValueError("atom2vperm: The bond vector set is not closed!")
             
        
        
        Pv.append(pv)
        Sv.append(sv)
    
    return Pv, Sv 
    
    
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
    r0: float
        The reference length.
    offset: float
        The offset of each coordinate function.
        
    """

    def __init__(self, n, a, r0 = 0.0, offset = 0.0 ):
        """

        Parameters
        ----------
        n : integer
            The number of atoms
        a : float
            The exponential length parameter.
        r0 : float, optional
            A reference length.
        offset : float, optional
            Offset of each coordinate function. 
            
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
        self.r0 = r0 
        self.offset = offset 
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
                yij = adf.exp(-(rij - self.r0) / self.a) - self.offset
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
    
class BondVectorR(nitrogen.dfun.DFun):
    """
    Bond vector function for linear distance
    
    ..  math ::
        
        v_{ij} = X_j - X_i 
    
    Attributes
    ----------
    n : integer
        The number of atoms.
    Vij : list of (2,)
        The bond pairs 
        
    """

    def __init__(self, n, Vij):
        """

        Parameters
        ----------
        n : integer
            The number of atoms
        Vij : list of (2,)
            The bond pairs 
        """
        
        if n < 2 :
            raise ValueError("There must be at least 2 atoms.")
            
        # The number of inputs is 3*n, the number of 
        # Cartesian coordinates 
        nx = 3*n 
        
        # The number of outputs is 3 x the number of 
        # vectors 
        nv = len(Vij)
        nf = 3 * nv 
        
        super().__init__(self._fvec, nf = nf, nx = nx,
                         maxderiv = None, zlevel = None)
        
        self.n = n
        self.Vij = Vij 
        
        return 
    
    
    def _fvec(self, X, deriv = 0, out = None, var = None):
        """ evaluation function """
        
        x = nitrogen.dfun.X2adf(X, deriv, var)
        
        y = [] 
        for vij in self.Vij: 
            A,B = vij 
            
            # v_ij = XB - XA
            
            for j in range(3): # x, y, z coordinates
            
                y.append(x[3*B+j] - x[3*A+j])
        
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
        
def fitSimplePIPDipole(X,D,P,Vij, yfun,vfun, degree,Xscale):
    """
    Fit a permutationally invariant polynomial surface for 
    a dipole moment function.

    Parameters
    ----------
    X : (3*n,N) ndarray
        The Cartesian coordinates of `n` particles at `N` sampling points.
    D : (nd,3,N) ndarray
        The dipole moment derivative array in the same frame as `X` at each point.
    P : list
        The permutation elements. Each element of `P` is a
        permutation of ``[0,1,2,...,n-1]``. 
    Vij : list of (2,)
        The list of bond vector atom-pairs, e.g., ``[(0,1), (0,2)]``.
    yfun : DFun
        The internuclear coordinate function.
    vfun : DFun
        The bond vector function.
    degree : integer
        The polynomial degree.
    Xscale : float
        The Cartesian length scale. Its powers will be used to scale
        derivatives to the value units.

    Returns
    -------
    p : (nvec, nterms) ndarray
        The expansion coefficients of each bond-vector expansion.
    res : ndarray
        The scaled residuals.
    D_function : DFun
        The fitted dipole function, D(X).
    
    Notes
    -----
    Derivatives up to arbitrary order can be fitted.
    
    See Also
    --------
    InternuclearExp : internuclear function for exponentially scaled distance
    InternuclearR : internuclear function for linear distance 

    """
    
    #####################################
    # A simple PIP fitting routine for dipole
    # moments using small molecular permutation groups
    #
    natoms = X.shape[0] // 3 # the number of atoms 
    ny = (natoms*(natoms-1)) // 2 # The number of internuclear y coordinates
    nv = len(Vij) # The number of bond vectors 
    
    #
    # The `y` variables are the n*(n-1)//2 functions
    # of the internuclear distances. This can be any
    # 'radial' function, e.g. r12, exp(-r12/a), etc.,
    # which is passed by the caller.
    #
    # The bond vectors are given by the (2,) elements of 
    # Vij. For element (i,j), the bond vector is
    # v = Xj - Xi, i.e. it points from atom i 
    # to atom j
    #
    # Atomic permutations permute bond vectors into 
    # each other. The set of vectors in Vij must be 
    # closed (up to negative signs)
    #
    
    # Calculate the y-index permutations from the 
    # atomic permutations
    Py = atom2yperm(P)
    
    # Calculate the bond-vector permutations
    # and sign coefficients
    Pv,Sv = atom2vperm(P, Vij)
    
    nP = len(P) # The number of permutation operations 
    
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
    
    #
    # We now figure out the symmetric projection of the 
    # products of y-monomials with bond basis vectors
    # 
    # Each permutation operator maps a monomial-vector product 
    # to another monomial-vector product, with a possible change in 
    # sign.
    #
    # For each monomial-vector product, 
    # find the lexical maximum of its invariant projection.
    # This is the "representative monomial" or "representative index"
    # We also need to keep track of the sign of the monomial-vector
    # pair relative to its representative pair.
    #
    # The lexical value is defined first by the 
    # bond basis vector, and then by the monomial (using standard ordering)
    #
    #  
    
    nck = adf.ncktab(ny+degree-1,min(ny,degree-1)) # A binomial coefficient table
    
    # unique_term_pos and unique_term_vector
    # will keep track of which 
    # representative monomial-vector pair 
    # each monomial-vector product maps to
    # 
    # unique_term_sign keeps track of the relative
    # sign of a given monomial-vector pair and its
    # representative
    #
    
    unique_term_pos = np.zeros((nv,nt,), dtype = np.uint32)
    unique_term_vector = np.zeros((nv,nt,), dtype = np.uint32)
    unique_term_sign = np.zeros((nv,nt,), dtype = np.int32)
    
    def inverse_perm(p):
        # Calculate the inverse permutation 
        ip = [p.index(val) for val in range(len(p))]
        return ip 
    
    for k in range(nv):  # For bond vector k
        for i in range(nt): # For monomial i of this expansion
            
            pi = pows[i,:] # The monomial's power index
            
            
            # Initialize the representative monomial power indices,
            # vector label `k`, and vector sign
            rep_idx = pi 
            rep_k   = k 
            rep_sign = +1
            
            # Find the lexical maximum of
            # all of its permutations 
            # 
            # For each permutation
            for opi in range(nP):
                py = Py[opi] # The permutation of the y variables
                pv = Pv[opi] # The permutation of the bond-vectors
                sv = Sv[opi] # the sign change of the bond-vectors
                
                
                # Calculate the permutation image
                # (Note: in fitSimplePIP, we do not bother to 
                #  use the correct inverse permutation look-up because
                #  the representative monomial is invariant. Here, we need
                #  to make sure the correct monomial-vector pair are transformed
                #  together.)
                perm_pi = pi[inverse_perm(py)] # The permuted monomial
                perm_k  = pv[k]
                perm_sign = sv[k]
                
                # Now update the representative monomial-vector pair 
                #
                if perm_k > rep_k:
                    # The current image is lexically greater
                    rep_idx = perm_pi 
                    rep_k = perm_k 
                    rep_sign = perm_sign 
                elif perm_k == rep_k:
                    # Sort by monomial now
                    if np.all(lexical_max(rep_idx, perm_pi) == perm_pi):
                        # The image is lexically greater 
                        # update 
                        rep_idx = perm_pi 
                        rep_k = perm_k 
                        rep_sign = perm_sign 
                else:
                    # perm_k < rep_k
                    # Keep the current representative 
                    pass 
            
            # The representative term is rep_idx, rep_k, and rep_sign.
            # 
            unique_term_pos[k,i] = adf.idxpos(rep_idx, nck)
            unique_term_sign[k,i] = rep_sign 
            unique_term_vector[k,i] = rep_k 
    
    # The "1-D" term ordering will by vector first, 
    # then each monomial in the expansion for that vector
    #
    unique_term_combined = unique_term_vector * nt + unique_term_pos 
    unique_term_combined = np.reshape(unique_term_combined, (-1,))
    
    unique_term_sign = np.reshape(unique_term_sign, (-1,))
    
    
    # Get the sorted list of unique, representative monomial-vector pairs
    rep_terms = np.sort(np.unique(unique_term_combined))
    nr = len(rep_terms) # unique terms after projection
    
    print(f"There are {ny:d} pair-coordinates.")
    print(f"For degree = {degree:d}, there are {nt:d} monomials.")
    print(f"There are {nv:d} bond-vectors.")
    print(f"There are {nv:d}*{nt:d} = {nv*nt:d} monomial-vector pairs.")
    print(f"After projection, there are {nr:d} invariant terms.")
    print("")
    print("Calculating least-squares matrix...", end = "")
    
    # `rep_terms` will serve as the order of the columns of the 
    # least-squares array. Figure out which column each
    # monomial maps to
    monomial_map = np.searchsorted(rep_terms, unique_term_combined)
    
    # Create the least-squares array
    # Initialize to zero 
    
    data_nd = D.shape[0] # The number of Cartesian derivatives supplied by caller
    npoints = D.shape[2] # The number of geometries 
    data_deg = nitrogen.dfun.infer_deriv(data_nd, 3*natoms) # The derivative order
    # of the data
    
    C = np.zeros((data_nd,3,npoints,nr))
    
    
    # Calculate the value/derivatives of each monomial term of the 
    # surface expansion w.r.t. X coordinates
    Z_of_y = nitrogen.dfun.PowerExpansionTerms(degree, np.zeros((ny,)))
    Z_of_X = yfun ** Z_of_y 

    ymonomials = Z_of_X.f(X, deriv = data_deg)  # shape (data_nd, nt, npoints)
    
    #################################################
    # We also need to calculate the derivative arrays of the bond vectors
    dv = vfun.f(X, deriv = data_deg) # (nd, 3*nv, npoints)
    #
    ################################################
    #
    # ymonomials contains the Cartesian derivatives of each monomial in the 
    # polynomial expansion
    #
    # dv contains the Cartesian derivatives of each bond basis vector
    #
    ################################################
    
    leib_nck = adf.ncktab(data_deg + 3*natoms, min(3*natoms, data_deg))
    leib_idx = adf.idxtab(data_deg, 3*natoms)
    leib_prod = lambda a,b : adf.mvleibniz(a, b, data_deg, 3*natoms, leib_nck, leib_idx)
    
    # Accumulate each monomial into the correct
    # final fitting term
    kr_idx = 0 
    
    for k in range(nv):
        for r in range(nt):
            # Add this monomial-vector pair to the appropriate columns of C
            #
            # Fetch the derivative arrays of the y-monomial
            # and bond-vector components separately
            ymon = ymonomials[:,r,np.newaxis,:] # (nd, 1, npoints)
            vxyz = dv[:, 3*k:3*(k+1), :]    # (nd, 3, npoints) 
            sign = unique_term_sign[kr_idx]
            
            
            # Calculate their product derivatives using the 
            # Leibniz product rule
            #
            mon_vec_prod = leib_prod(ymon, vxyz) # (nd, 3, npoints)
            
            # Accumulate the final derivative array in the least-squares matrix 
            C[:,:,:,monomial_map[kr_idx]] += sign * mon_vec_prod 
            
            kr_idx += 1 
            
    # Multiply Cartesian derivatives by approriate length scale
    # passed as Xscale 
    #
    Dcopy = np.copy(D) # A copy of D, which we can modify 
    
    for deg in range(1,data_deg+1):
        start = nitrogen.dfun.nderiv(deg-1, 3*natoms)
        stop = nitrogen.dfun.nderiv(deg, 3*natoms)
        
        C[start:stop] *= Xscale**deg 
        Dcopy[start:stop] *= Xscale**deg

    # Now reshape C 
    # The original shape is (nd,3,npoints,nr)
    # 
    C = C.reshape((data_nd*3*npoints, nr))
    # and the fit data, D.
    # The original shape is (nd,3,npoints)
    b = Dcopy.reshape((data_nd*3*npoints,))
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
    # full list of individual monomial-vector pairs, 
    # being sure to account for the relative
    # sign.
    #
    pfull = np.zeros((nv*nt,))
    for kr in range(nv*nt):
        pfull[kr] = unique_term_sign[kr] * p[monomial_map[kr]]
    
    pfull = pfull.reshape((nv, nt))
    
    ########################
    # Results        
    rmse = np.sqrt(np.average(res**2)) # Total scaled RMS residual
    print("--------------------------------------")
    print(f"Total scaled rmse = {rmse:.3f}")
    print("--------------------------------------")
    # Partition RMS error by derivative order 
    res = res.reshape((data_nd,3*npoints))
    
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
    
    D_function = BondVectorPIP(yfun, vfun, pfull)
    
    return pfull, res, D_function 
    

class BondVectorPIP(nitrogen.dfun.DFun):
    """
    
    A general PIP expansion for dipole or other vector
    functions.
    
    """
    
    def __init__(self, yfun, vfun, p):
        """
        

        Parameters
        ----------
        yfun : DFun
            The internuclear distance function.
        vfun : DFun
            The bond vector function.
        p : (nvec,n) ndarray
            The expansion coefficients of each bond vector function.

        """
        
        
        nf = 3 # Three dipole components
        nx = vfun.nx # The total number of Cartesian components
        
        
        super().__init__(self._fbondvec, nf = nf, nx = nx,
                         maxderiv = None, zlevel = None)
        
        # Create the PowerExpansion DFun's for each
        # bond vector coefficient 
        
        d = p.T.copy() # (number of expansion terms, number of vectors)
        ny = yfun.nf   # The number of internuclear distance functions
        
        
        # qfun ...  (X --> y) ** (y --> q) 
        #
        # The q(X) function evaluates the coefficients of each bond vector
        #
        self.qfun = yfun ** nitrogen.dfun.PowerExpansion(d, np.zeros((ny,)) )
        
        
        self.yfun = yfun 
        self.vfun = vfun 
        self.nvec = p.shape[0] # The number of bond vectors 
        
        if (self.nvec != self.vfun.nf // 3):
            raise ValueError("p and vfun have mis-matched numbers of vectors")
        
        return 
    
    
    def _fbondvec(self, X, deriv = 0, out = None, var = None):
        """ evaluation function """
        
        # 
        # We calculate q(X), the coefficients of each bond-vector
        # as well as the bond-vectors themselves.
        #
        nd,nvar = nitrogen.dfun.ndnvar(deriv, var, self.nx)
        
        # Then we use the Leibniz product rule to 
        # multiply their derivative arrays
        #
        dq = self.qfun.f(X, deriv = deriv, out = None, var = var)  # (nd, nv, ...)
        dv = self.vfun.f(X, deriv = deriv, out = None, var = var)  # (nd, nv*3, ...)
        
        nv = self.nvec # The number of bond vectors 
        
        dq = dq[:,:,np.newaxis,...] # (nd, nv, 1, ...)
        dv = np.reshape(dv, (nd, nv, 3) + dv.shape[2:]) # (nd, nv, 3, ...)
        
        
        
        leib_nck = adf.ncktab(deriv + nvar, min(nvar, deriv))
        leib_idx = adf.idxtab(deriv, nvar)

        # Note broadcasting:
        # (nd, nv, 1, ...) x (nd, nv, 3, ...)
        # --> (nd, nv, 3, ...)
        #
        expansion_vec_prod = adf.mvleibniz(dq, dv, deriv, nvar, leib_nck, leib_idx)
            
        # The total vector function is the sum over the `nv` basis vectors 
        V = np.sum(expansion_vec_prod, axis = 1) # (nd, 3, ...)
        
        return V 
    