"""
nitrogen.vpt.distho

Distributed Gaussian/Harmonic Oscillator Basis Sets

"""

import numpy as np 
import nitrogen.autodiff.forward as adf 
import nitrogen as n2 
from scipy.linalg import eigh
import time 
import matplotlib.pyplot as plt 

class DistHOBasis:
    """
    
    This class implements basis sets comprising distributed
    sets of multidimensional harmonic oscillator basis functions.

    Each center is defined by a position :math:`a` in a :math:`n`-dimensional
    coordinate space :math:`x` and a normal-mode transformation matrix :math:`W`.
    
    The vacuum basis functions at each center are defined as 

    ..  math::
        
        f(x; a, W) = \\exp[-(x-a)^T A (x-a)],
        
    where
    
    ..  math::
        
        A = \\frac{1}{2} W^T W
    
    The conventional reduced dimensionless normal coordinates are
    
    ..  math::
        
        q = W (x - a)
        
    in terms of which the basis functions are simply :math:`\\exp(-\\frac{1}{2} q^Tq)`.
    
    The excitation functions are
    
    ..  math ::
        
        \\langle x \\vert n \\rangle &= f(x; a, W, n) 
        
        &= \\exp(-\\frac{1}{2}q^T q) \\prod_i \\phi_{n_i}(q_i)
        
        \\phi_n(q) &= \\frac{1}{\\sqrt{2^{n} n !}} H_n(q)
        
    where :math:`H_n` are the Hermite polynomials. 
    
    Attributes
    ----------
    N : integer
        The number of centers 
    n : integer
        The number of coordinates.
    a : (N,n) ndarray
        The basis centers
    W : (N,n,n) ndarray
        The basis normal-mode transformations
    A : (N,n,n) ndarray
        The value of :math:`\\frac{1}{2} W^T W`
    S : (N,N) ndarray
        The vacuum overlap integrals.
    c : (N,N,n) ndarray
        The product centers
    C : (N,N,n,n) ndarray
        The product coefficient arrays
    expd : (N,N,n) ndarray
        The product pre-factors, :math:`\\exp(-d)`.
    STTP : (N,N,3,n,n) ndarray
        The :math:`S`, :math:`T`, and :math:`T'` recursion coefficient arrays.
    rrp : (N,N,3,n) ndarray
        The :math:`r` and :math:`r'` recursion coefficient vectors.
    XXp : (N,N,2,n,n) ndarray
        The :math:`X` and :math:`X'` operator normal-order transformation coefficients.
    YYp : (N,N,2,n,n) ndarray
        The :math:`Y` and :math:`Y'` operator normal-order transformation coefficients.
    
    """
    
    def __init__(self, a, W):
        """
        

        Parameters
        ----------
        a : (N,n) array_like
            The center positions
        W : (N,n,n) array_like
            The normal-mode transformations

        """
        
        a = np.array(a)
        N,n = a.shape 
        W = np.array(W) 
        
        ######################
        # Calculate single-center derived quantities
        #
        A = np.zeros((N,n,n))
        iW = np.zeros((N,n,n))
        
        for i in range(N):
            A[i] = 0.5 * W[i].T @ W[i] 
            iW[i] = np.linalg.inv(W[i])
        
        ######################
        # Calculate basic product information
        # for basis set
        #
        S = np.zeros((N,N))          
        c = np.zeros((N,N,n))
        C = np.zeros((N,N,n,n))
        iC = np.zeros((N,N,n,n))
        expd = np.zeros((N,N,n))
        v = np.zeros((N,N,n))
        
        # AiCB = np.zeros((N,N,n,n))  # A @ iC @ B for each pair 
        
        STTp = np.zeros((N,N,3,n,n)) # The S, T, and Tp matrices for each <A|B> pair
        rrp = np.zeros((N,N,2,n)) # The r, rp vectors for each <A|B> pair 
        
        XXp = np.zeros((N,N,2,n,n))
        YYp = np.zeros((N,N,2,n,n))
        
        Vp = np.zeros((N,N,n,n))
        
        # y = 2*v 
        
        for i in range(N):
            for j in range(i + 1):
                
                cij, Cij, dij, iCij = product_gaussian(a[i], A[i], a[j], A[j])
                
                expdij = np.exp(-dij) 
                normCij = norm(Cij)
                
                vij = A[i] @ (cij - a[i]) # = - A[j] @ (cij - a[j]) = -vji
                
                # Evaluate the vacuum overlap integral 
                Sij =  expdij * normCij
                
                # AiCB_ij = A[i] @ iCij @ A[j] 
                
                expd[i,j] = expdij
                expd[j,i] = expdij
                
                S[i,j] = Sij 
                S[j,i] = Sij 
                
                c[i,j] = cij 
                c[j,i] = cij 
                
                C[i,j] = Cij 
                C[j,i] = Cij 
                
                iC[i,j] = iCij 
                iC[j,i] = iCij 
                
                v[i,j] = vij 
                v[j,i] = -vij # note sign! 
                
                # AiCB[i,j] = AiCB_ij 
                # AiCB[j,i] = AiCB_ij.T  # note transpose! 
                


        # Calculate recursion coefficients        
        # and normal-ordered expansion coefficients 
        
        for i in range(N):
            for j in range(N):
                s,t,tp,r,rp = calcTwoCenterRecursion(a[i], W[i], iW[i], a[j], W[j], iW[j])
                
                STTp[i,j,0] = s 
                STTp[i,j,1] = t
                STTp[i,j,2] = tp
                rrp[i,j,0] = r 
                rrp[i,j,1] = rp 
                
                #
                # X = sqrt[1/2] W'^-1 @ S'  (Note S' = S^T)
                # X' = sqrt[1/2] W^-1 @ S
                #
                XXp[i,j,0] = np.sqrt(0.5) * iW[j] @ s.T 
                XXp[i,j,1] = np.sqrt(0.5) * iW[i] @ s
                
                YYp[i,j,0] = -np.sqrt(0.5) * W[j].T @ s.T 
                YYp[i,j,1] = +np.sqrt(0.5) * W[i].T @ s 
                
                Vp[i,j] = 0.5 * (W[j] @ iW[i] + iW[j].T @ W[i].T)
                
                # y = np.sqrt(0.5) * W.T @ r  = 2 * v 
     
         
        self.v = v 
        # self.AiCB = AiCB 
        
        self.n = n 
        self.N = N 
        self.a = a 
        self.W = W
        self.iW = iW 
        self.A = A 
        
        self.S = S 
        self.c = c 
        self.C = C 
        self.iC = iC 
        self.expd = expd
        
        self.STTp = STTp 
        self.rrp = rrp 
        self.XXp = XXp 
        self.YYp = YYp 
        self.Vp = Vp 
        
        return 
    
    def calcSigma(self,nex):
        """
        Calculate overlaps 
        
        Parameters
        ----------
        nex : integer
            The maximum excitation number
        """
        
        if nex > 2:
            raise ValueError("Only nex = 0,1,2 are hand-coded")
        
        
        N = self.N 
        n = self.n 
        
        if nex == 0: 
            nb = 1 
        elif nex == 1:
            nb = 1 + n 
        elif nex == 2:
            nb = ( (n+1)*(n+2) ) // 2
        
        Sigma = np.zeros((N,nb,N,nb)) 
        
        for I in range(N):
            for J in range(N):
                # pair (I,J) 
                STTp = self.STTp[I,J]
                rrp = self.rrp[I,J] 
                SIJ = self.S[I,J] 
                
                s00 = np.ones((1,1)) 
                
                if nex >= 1:
                    s01 = _overlap01(STTp, rrp)
                    s10 = _overlap10(STTp, rrp) 
                    s11 = _overlap11(STTp, rrp) 
                
                if nex >= 2:
                    s02 = _overlap02(STTp, rrp)
                    s12 = _overlap12(STTp, rrp)
                    s22 = _overlap22(STTp, rrp)
                    s21 = _overlap21(STTp, rrp)
                    s20 = _overlap20(STTp, rrp)
                
                if nex == 0:
                    SigmaIJ = s00 
                elif nex == 1:
                    SigmaIJ = np.block([[s00, s01],
                                        [s10, s11]])
                elif nex == 2:
                    SigmaIJ = np.block([[s00, s01, s02],
                                        [s10, s11, s12],
                                        [s20, s21, s22]])
                
                Sigma[I,:,J,:] = SIJ * SigmaIJ 
        
        return Sigma 
    
    
    def normal_order_coordinate0(self, val):
        """
        Return the normal-ordered expression 
        for a scalar degree-0 expansion at 
        each pair center.

        Parameters
        ----------
        val : (N,N)
            The value at each pair center.

        Returns
        -------
        d0 : (N,N)
            The normal-ordered operator
            
        Notes
        -----
        Because this is a constant scalar, 
        `d0` is just a copy of `val`.
        
        """
        
        return val.copy() 
    
    def normal_order_coordinate1(self, val, jac) :
        """
        Return the normal-ordered expression 
        for a scalar degree-1 expansion at each 
        pair center.
        
        Parameters
        ----------
        val : (N,N)
            The value at each pair center.
        jac : (N,N,n)
            The gradient at each pair center.

        Returns
        -------
        d0 : (N,N)
            The constant :math:`d_0` value.
        da : (N,N,n)
            The :math:`d_a` vectors
        db : (N,N,n)
            The :math:`d_b` vectors 
            
        See Also
        --------
        operator_degree0 : evaluate matrix elements for degree-0 operator
        
        operator_degree1 : evaluate matrix elements for degree-1 operator 
        
        """
        
        N = self.N 
        n = self.n 
        
        d0 = self.normal_order_coordinate0(val)
        
        da = np.zeros((N,N,n))
        db = np.zeros((N,N,n))
        
        # 
        # The expansion is vac + jac @ (x-c)
        # 
        # (x-c) = X' beta + X alpha^dagger
        #
        # so the beta coefficients are jac @ X'
        # and the alpha coefficients are jac @ X 
        #
        
        for i in range(N):
            for j in range(N):
                
                X,Xp = self.XXp[i,j]
                
                da[i,j] += jac[i,j] @ X
                db[i,j] += jac[i,j] @ Xp 
        
        return d0, da, db 
    
    def normal_order_coordinate2(self, val, jac, hes):
        """
        Return the normal-ordered expression 
        for a scalar degree-2 expansion at each 
        pair center.
        
        Parameters
        ----------
        val : (N,N)
            The value at each pair center.
        jac : (N,N,n)
            The gradient at each pair center.
        hes : (N,N,n,n)
            The hessian at each pair center.

        Returns
        -------
        d0 : (N,N)
            The constant :math:`d_0` value.
        da : (N,N,n)
            The :math:`d_a` vectors
        db : (N,N,n)
            The :math:`d_b` vectors 
        Daa : (N,N,n,n)
            The :math:`D_{aa}` arrays.
        Dab : (N,N,n,n)
            The :math:`D_{ab}` arrays.
        Dbb : (N,N,n,n)
            The :math:`D_{bb}` arrays.
            
        See Also
        --------
        operator_degree0 : evaluate matrix elements for degree-0 operator
        
        operator_degree1 : evaluate matrix elements for degree-1 operator 
        
        operator_degree 2 : evaluate matrix elements for degree-2 operator 
        
        """
        
        N = self.N 
        n = self.n 
        
        d0, da, db = self.normal_order_coordinate1(val, jac)
        
        Daa = np.zeros((N,N,n,n))
        Dab = np.zeros((N,N,n,n))
        Dbb = np.zeros((N,N,n,n))
        
        # 
        # The second-order contribution is 
        # 
        # 0.5 * (x-c)^T @ hes @ (x-c)
        #
        # 
        for i in range(N):
            for j in range(N):
                
                X,Xp = self.XXp[i,j]
                iC = self.iC[i,j]
                
                F = 0.5 * hes[i,j] 
                
                Daa[i,j] += X.T @ F @ X 
                Dab[i,j] += X.T @ (F.T + F) @ Xp 
                Dbb[i,j] += Xp.T @ F @ Xp 
                
                # Commutator contribution to degree-0
                d0[i,j] += 0.5 * np.sum(F * iC)  # tr[F @ iC] , iC is symmetric 
       
        return d0, da, db, Daa, Dab, Dbb 
    
    def normal_order_phi3(self, phi3):
        """
        Return the normal-ordered expression for a cubic
        coordinate expansion
        
        ..  math:: 
            
            \\frac{1}{6} \\sum_{ijk} \\phi_{ijk} (x-c)_i (x-c)_j (x-c)_k

        Parameters
        ----------
        phi3 : (N,N,n,n,n)
            The cubic force field parameters

        Returns
        -------
        da : (N,N,n)
            The :math:`d_a` vectors
        db : (N,N,n)
            The :math:`d_b` vectors 
        Daaa : (N,N,n,n,n)
            The :math:`D_{aaa}` arrays.
        Daab : (N,N,n,n,n)
            The :math:`D_{aab}` arrays.
        Dabb : (N,N,n,n,n)
            The :math:`D_{abb}` arrays.
        Dbbb : (N,N,n,n,n)
            The :math:`D_{bbb}` arrays.

        """
        
        N = self.N 
        n = self.n 
        
        da = np.zeros((N,N,n))
        db = np.zeros((N,N,n))
        
        Daaa = np.zeros((N,N,n,n,n))
        Daab = np.zeros((N,N,n,n,n))
        Dabb = np.zeros((N,N,n,n,n))
        Dbbb = np.zeros((N,N,n,n,n))
        
        # Pre-optimize the tensor contractions
        U = np.random.rand(n,n,n)
        R = np.random.rand(n,n)
        path1 = np.einsum_path('ijk,il,jm,kn->lmn',U,R,R,R)[0]
        path2 = np.einsum_path('ijk,il,jm,kn->nlm',U,R,R,R)[0]
        path3 = np.einsum_path('ijk,il,jm,kn->mnl',U,R,R,R)[0]
        
        path4 = np.einsum_path('ijk,il,jm,kn,mn->l',U,R,R,R,R)[0]
        path5 = np.einsum_path('ijk,il,jm,kn,mn->l',U,R,R,R,R)[0]
        path6 = np.einsum_path('ijk,il,jm,kn,ln->m',U,R,R,R,R)[0]
        
        
        for i in range(N):
            for j in range(N):
                
                X,Xp = self.XXp[i,j]
                Vp = self.Vp[i,j] 
                phi = phi3[i,j] 
                
                Daaa[i,j] += np.einsum('ijk,il,jm,kn->lmn', phi, X,  X,  X,  optimize = path1)
                Dbbb[i,j] += np.einsum('ijk,il,jm,kn->lmn', phi, Xp, Xp, Xp, optimize = path1)
                
                Daab[i,j] += np.einsum('ijk,il,jm,kn->lmn', phi, X,  X,  Xp, optimize = path1)
                Daab[i,j] += np.einsum('ijk,il,jm,kn->nlm', phi, X,  Xp, X,  optimize = path2)
                Daab[i,j] += np.einsum('ijk,il,jm,kn->mnl', phi, Xp, X,  X,  optimize = path3)
                
                da[i,j] += np.einsum('ijk,il,jm,kn,mn->l', phi, X, Xp, X, Vp, optimize = path4)
                da[i,j] += np.einsum('ijk,il,jm,kn,lm->n', phi, Xp, X, X, Vp, optimize = path5)
                da[i,j] += np.einsum('ijk,il,jm,kn,ln->m', phi, Xp, X, X, Vp, optimize = path6)
                
                Dabb[i,j] += np.einsum('ijk,il,jm,kn->lmn', phi, X, Xp, Xp, optimize = path1)
                Dabb[i,j] += np.einsum('ijk,il,jm,kn->mnl', phi, Xp, X, Xp, optimize = path3)
                Dabb[i,j] += np.einsum('ijk,il,jm,kn->nlm', phi, Xp, Xp, X, optimize = path2)
                
                db[i,j] += np.einsum('ijk,il,jm,kn,lm->n', phi, Xp, X, Xp, Vp, optimize = path5)
                db[i,j] += np.einsum('ijk,il,jm,kn,mn->l', phi, Xp, Xp, X, Vp, optimize = path4)
                db[i,j] += np.einsum('ijk,il,jm,kn,ln->m', phi, Xp, Xp, X, Vp, optimize = path6)
                
                
        # 
        # 1/6 factor from definition of phi_ijk
        #
        da /= 6 
        db /= 6 
        Daaa /= 6
        Daab /= 6
        Dabb /= 6
        Dbbb /= 6
        
        return da, db, Daaa, Daab, Dabb, Dbbb 
    
    def normal_order_deriv2(self, G):
        """
        Normal order the derivative operator :math:`G_{ij} \\partial_{x_i} \\partial_{x_j}`.
        (:math:`G` need not be symmetric.)

        Parameters
        ----------
        G : (N,N,n,n) ndarray
            The coefficient array at each pair-center.

        Returns
        -------
        d0 : (N,N)
            The constant :math:`d_0` value.
        da : (N,N,n)
            The :math:`d_a` vectors
        db : (N,N,n)
            The :math:`d_b` vectors 
        Daa : (N,N,n,n)
            The :math:`D_{aa}` arrays.
        Dab : (N,N,n,n)
            The :math:`D_{ab}` arrays.
        Dbb : (N,N,n,n)
            The :math:`D_{bb}` arrays.
            

        """
        
        N = self.N 
        n = self.n 
        
       
        d0 = np.zeros((N,N))
        
        da = np.zeros((N,N,n))
        db = np.zeros((N,N,n))
        
        Daa = np.zeros((N,N,n,n))
        Dab = np.zeros((N,N,n,n))
        Dbb = np.zeros((N,N,n,n))
        
        for i in range(N):
            for j in range(N):
                
                Y,Yp = self.YYp[i,j]
                y = 2 * self.v[i,j] 
                A = self.A[i]
                B = self.A[j] 
                iC = self.iC[i,j]
                
                F = G[i,j]
                
                Daa[i,j] += Y.T @ F @ Y 
                Dab[i,j] += Y.T @ (F.T + F) @ Yp 
                Dbb[i,j] += Yp.T @ F @ Yp
            
                da[i,j] += y @ (F + F.T) @ Y 
                db[i,j] += y @ (F + F.T) @ Yp 
                
                d0[i,j] += y @ F @ y  
                d0[i,j] += -2 * np.sum(F * (A@iC@B))
                # - 2 * tr(G.T @ (A @ iC @ B)) (from commutator of b...a term)
       
        return d0, da, db, Daa, Dab, Dbb 
    
    def normal_order_derivcoord21(self, G1):
        """
        Normal order the second derivative operator 
        :math:`G^{(1)}_{i,jk} \\partial_{x_j} (x-c)_i \\partial_{x_k}`

        Parameters
        ----------
        G1 : (N,N,n,n,n) ndarray
            The operator expansion coefficients.

        Returns
        -------
        d0 : (N,N)
            The constant :math:`d_0` value.
        da : (N,N,n)
            The :math:`d_a` vectors
        db : (N,N,n)
            The :math:`d_b` vectors 
        Daa : (N,N,n,n)
            The :math:`D_{aa}` arrays.
        Dab : (N,N,n,n)
            The :math:`D_{ab}` arrays.
        Dbb : (N,N,n,n)
            The :math:`D_{bb}` arrays.
        Daaa : (N,N,n,n,n)
            The :math:`D_{aaa}` arrays.
        Daab : (N,N,n,n,n)
            The :math:`D_{aab}` arrays.
        Dabb : (N,N,n,n,n)
            The :math:`D_{abb}` arrays.
        Dbbb : (N,N,n,n,n)
            The :math:`D_{bbb}` arrays.
            
        """
        
        
        N = self.N 
        n = self.n 
        
        d0 = np.zeros((N,N))
        
        da = np.zeros((N,N,n))
        db = np.zeros((N,N,n))
        
        Daa = np.zeros((N,N,n,n))
        Dab = np.zeros((N,N,n,n))
        Dbb = np.zeros((N,N,n,n))
        
        Daaa = np.zeros((N,N,n,n,n))
        Daab = np.zeros((N,N,n,n,n))
        Dabb = np.zeros((N,N,n,n,n))
        Dbbb = np.zeros((N,N,n,n,n))
        
        # Pre-optimize the tensor contractions
        U = np.random.rand(n,n,n)
        R = np.random.rand(n,n)
        r = np.random.rand(n)
        
        path1 = np.einsum_path('ijk,jm,il,kn->mln', U, R, R, R)[0] 
        path2 = np.einsum_path('ijk,j,il,kn->ln', U, r, R, R)[0]
        path3 = np.einsum_path('ijk,jm,il,k->ml', U, R, R, r)[0] 
        path4 = np.einsum_path('ijk,j,il,k->l', U, r, R, r)[0] 
        
        path5 = np.einsum_path('ijk,j,il,kn->nl', U, r, R, R)[0]
        path6 = np.einsum_path('ijk,j,il,kn,ln', U, r, R, R, R)[0] 
        
        path7 = np.einsum_path('ijk,jm,il,k->lm', U, R, R, r)[0] 
        path8 = np.einsum_path('ijk,jm,il,k,ml', U, R, R, r, R)[0] 
        
        path9 = np.einsum_path('ijk,jm,il,kn->lnm', U, R, R, R)[0] 
        path10 = np.einsum_path('ijk,jm,il,kn->nml', U, R, R, R)[0] 
        
        path11 = np.einsum_path('ijk,jm,il,kn,ml->n', U, R, R, R, R)[0] 
        path12 = np.einsum_path('ijk,jm,il,kn,ln->m', U, R, R, R, R)[0] 
        path13 = np.einsum_path('ijk,jm,il,kn,mn->l', U, R, R, R, R)[0] 
        
        for i in range(N):
            for j in range(N):
                
                X,Xp = self.XXp[i,j]
                Y,Yp = self.YYp[i,j]
                y = 2 * self.v[i,j] 
                Vp = self.Vp[i,j] 
                G = G1[i,j]   # G^(1)_i;jk
                
                # aaa, bbb
                Daaa[i,j] += np.einsum('ijk,jm,il,kn->mln', G, Y,  X,  Y , optimize = path1)
                Dbbb[i,j] += np.einsum('ijk,jm,il,kn->mln', G, Yp, Xp, Yp, optimize = path1)
                
                # 0bb, bb0
                # 0aa, aa0
                Dbb[i,j] += np.einsum('ijk,j,il,kn->ln', G, y,  Xp, Yp, optimize = path2)
                Dbb[i,j] += np.einsum('ijk,jm,il,k->ml', G, Yp, Xp, y,  optimize = path3)
                Daa[i,j] += np.einsum('ijk,y,il,kn->ln', G, y,  X,  Y,  optimize = path2)
                Daa[i,j] += np.einsum('ijk,jm,il,k->ml', G, Y,  X,  y, optimize = path3)
                
                # 0b0, 0a0
                db[i,j] += np.einsum('ijk,j,il,k->l', G, y, Xp, y, optimize = path4)
                da[i,j] += np.einsum('ijk,j,il,k->l', G, y, X,  y, optimize = path4) 
                
                # 0ab, ab0
                Dab[i,j] += np.einsum('ijk,j,il,kn->ln', G, y, X, Yp, optimize = path2) 
                Dab[i,j] += np.einsum('ijk,jm,il,k->ml', G, Y, Xp, y, optimize = path3) 
                
                # 0ba
                Dab[i,j] += np.einsum('ijk,j,il,kn->nl', G, y, Xp, Y, optimize = path5)
                d0[i,j] += np.einsum('ijk,j,il,kn,ln', G, y, Xp, Y, Vp, optimize = path6) 
                
                # ba0
                Dab[i,j] += np.einsum('ijk,jm,il,k->lm', G, Yp, X, y, optimize = path7)
                d0[i,j] += np.einsum('ijk,jm,il,k,ml', G, Yp, X, y, Vp, optimize = path8)
                
                # abb, bab, bba 
                Dabb[i,j] += np.einsum('ijk,jm,il,kn->mln', G, Y, Xp, Yp, optimize = path1)
                Dabb[i,j] += np.einsum('ijk,jm,il,kn->lnm', G, Yp, X, Yp, optimize = path9)
                Dabb[i,j] += np.einsum('ijk,jm,il,kn->nml', G, Yp, Xp, Y, optimize = path10) 
                
                db[i,j] += np.einsum('ijk,jm,il,kn,ml->n', G, Yp, X, Yp, Vp, optimize = path11)
                db[i,j] += np.einsum('ijk,jm,il,kn,ln->m', G, Yp, Xp, Y, Vp, optimize = path12)
                db[i,j] += np.einsum('ijk,jm,il,kn,mn->l', G, Yp, Xp, Y, Vp, optimize = path13)
                
                # aab, aba, baa
                Daab[i,j] += np.einsum('ijk,jm,il,kn->mln', G, Y, X, Yp, optimize = path1)
                Daab[i,j] += np.einsum('ijk,jm,il,kn->nml', G, Y, Xp, Y, optimize = path10)
                Daab[i,j] += np.einsum('ijk,jm,il,kn->lnm', G, Yp, X, Y, optimize = path9)
                
                da[i,j] += np.einsum('ijk,jm,il,kn,ln->m', G, Y, Xp, Y, Vp, optimize = path12)
                da[i,j] += np.einsum('ijk,jm,il,kn,mn->l', G, Yp, X, Y, Vp, optimize = path13)
                da[i,j] += np.einsum('ijk,jm,il,kn,ml->n', G, Yp, X, Y, Vp, optimize = path11)
                
        
        return d0, da, db, Daa, Dab, Dbb, Daaa, Daab, Dabb, Dbbb 
        
    
    def normal_order_derivcoord10(self, g):
        """
        Normal order the anti-hermitian derivative operator 
        :math:`\\partial_x^T g + g^T \\partial_x`.
        
        Parameters
        ----------
        g : (N,N,n) ndarray
            The derivative coefficients.

        Returns
        -------
        d0 : (N,N)
            The constant :math:`d_0` value.
        da : (N,N,n)
            The :math:`d_a` vectors
        db : (N,N,n)
            The :math:`d_b` vectors 
        

        """
        
        N = self.N 
        n = self.n 
        
        d0 = np.zeros((N,N))
        da = np.zeros((N,N,n))
        db = np.zeros((N,N,n))
        
        for i in range(N):
            for j in range(N):
                
                Y,Yp = self.YYp[i,j] 
                y = 2 * self.v[i,j] 
                
                d0[i,j] += 2 * np.dot(g[i,j],y)
                da[i,j] += 2 * g[i,j] @ Y 
                db[i,j] += 2 * g[i,j] @ Yp 
                
        return d0, da, db 
    
    def normal_order_derivcoord11(self, g, G):
        """
        Normal order the anti-Hermitian derivative operator 
        :math:`\\partial_x^T G^T (x-c) + (x-c)^T G \\partial_x`.

        Parameters
        ----------
        g : (N,N,n) ndarray
            The derivative constant coefficients.
        G : (N,N,n,n)
            The derivative linear expansion coefficients.
            (The first `n` axis is the linear expansion.
             The second `n` axis is the vibrational sum from :math:`T_{rv}`)

        Returns
        -------
        d0 : (N,N)
            The constant :math:`d_0` value.
        da : (N,N,n)
            The :math:`d_a` vectors
        db : (N,N,n)
            The :math:`d_b` vectors 
        Daa : (N,N,n,n)
            The :math:`D_{aa}` arrays.
        Dab : (N,N,n,n)
            The :math:`D_{ab}` arrays.
        Dbb : (N,N,n,n)
            The :math:`D_{bb}` arrays.

        """
        
        N = self.N 
        n = self.n 
        
        d0, da, db = self.normal_order_derivcoord10(g)
        
        Daa = np.zeros((N,N,n,n))
        Dab = np.zeros((N,N,n,n))
        Dbb = np.zeros((N,N,n,n))
        
        for i in range(N):
            for j in range(N):
                
                X,Xp = self.XXp[i,j]
                Y,Yp = self.YYp[i,j] 
                y = 2 * self.v[i,j] 
                Vp = self.Vp[i,j] 
                
                F = G[i,j]
                
                mbb = Xp.T @ F @ Yp
                Dbb[i,j] += mbb + mbb.T 
                
                maa = X.T @ F @ Y
                Daa[i,j] += maa + maa.T 
                
                Dab[i,j] += 2 * (X.T @ F @ Yp + Y.T @ F.T @ Xp)
                
                d0[i,j] += np.sum( F.T * (Y @ Vp.T @ Xp.T + Yp @ Vp @ X.T) )
                
                da[i,j] += 2 * X.T @ F @ y
                db[i,j] += 2 * Xp.T @ F @ y
        
        return d0, da, db, Daa, Dab, Dbb 
                
def product_gaussian(a,A,b,B):
    """
    Calculate the product Gaussian 
    
    ..  math::
        
        \\exp[-(x-a)^T A (x-a)] \\exp[-(x-b)^T B (x-b)] = 
        
        \\exp[-d] \\exp[-(x-c)^T C (x-c)] 

    Parameters
    ----------
    a : (n,) ndarray
        The first center position
    A : (n,n) ndarray
        The first center coefficient array 
    b : (n,) ndarray
        The second center position
    B : (n,n) ndarray
        The second center coefficient array 

    Returns
    -------
    c : (n,) ndarray
        The product center
    C : (n,n) ndarray
        The product coefficient array
    d : float
        The pre-factor argument
    iC : (n,n) ndarray
        The inverse of `C`.

    """
    
    C = A + B 
    iC = np.linalg.inv(C) 
    
    v = A @ a + B @ b
    c = iC @ v
    
    d = np.dot(a,A @ a) + np.dot(b,B @ b) - np.dot(v, c)
    
    return c, C, d, iC 

def norm(A):
    """
    Evaluate the integral of a general Gaussian
    
    ..  math ::
        
        \\int dx \\exp[-x^T A x] = \\frac{\\pi^{n/2}}{\\mathrm{det} A ^{1/2} }

    Parameters
    ----------
    A : (n,n) ndarray
        The coefficient matrix.

    Returns
    -------
    float 
        The integral

    """
    
    n = len(A)
    
    return np.sqrt( np.pi ** n / np.linalg.det(A) )

def calcTwoCenterRecursion(a,W,iW,ap,Wp,iWp):
    """
    Calculate the recursion coefficients for 
    a two-center pair

    Parameters
    ----------
    a : (n,) ndarray
        The bra center position
    W : (n,n) ndarray
        The bra normal-mode array
    iW : (n,n) ndarray
        The inverse of `W`.
    ap : (n,) ndarray
        The ket center position
    Wp : (n,n) ndarray
        The ket normal-mode array 
    iWp : (n,n) ndarray
        The inverse of `Wp`.

    Returns
    -------
    S : (n,n) ndarray
        The :math:`S` array 
    T : (n,n) ndarray
        The :math:`T` array 
    Tp : (n,n) ndarray
        The :math:`T'` array 
    r : (n,) ndarray
        The :math:`r` vector 
    rp : (n,) ndarray
        The :math:`r'` vector 
        
    Notes
    -----
    
    The recursion relationships for two-center harmonic oscillator
    basis sets can be derived following Fernandez and Tipping,
    JCP 91, 5505 (1989). Note that there are slight sign-convention
    and labeling differences from that reference here.
    
    The overlap integrals satisfy 
    
    ..  math:: 
        
        \\langle m_i + 1 \\vert \\vert \\cdots \\rangle &= 
        \\frac{1}{\\sqrt{m_i+1}} \\left[ r_i \\langle \\cdots 
        \\vert \\vert \\cdots \\rangle + \\sum_j S_{ij} \\sqrt{n_j}
        \\langle \\cdots \\vert \\vert n_j - 1\\rangle + T_{ij}\\sqrt{m_j}
        \\langle m_j - 1 \\vert \\vert \\cdots \\rangle \\right]
        
        \\langle \\cdots \\vert \\vert n_i + 1 \\rangle &= 
        \\frac{1}{\\sqrt{n_i+1}} \\left[ r'_i \\langle \\cdots \\vert\\vert
        \\cdots\\rangle + \\sum_j T_{ij}' \\sqrt{n_j} \\langle \\cdots \\vert
        \\vert n_j -1 \\rangle + S_{ij}' \\sqrt{m_j} \\langle m_j - 1 \\vert
        \\vert \\cdots \\rangle \\right]

    where :math:`S' = S^T`.
    
    Swapping the bra and ket leaves transforms :math:`S` to its transpose (i.e.,
    it swaps :math:`S` and :math:`S'`), swaps *and* transposes :math:`T` and :math:`T'`,
    and swaps :math:`r` and :math:`r'`.
    
    """
    
    R = Wp @ iW 
    iR = W @ iWp 
    
    Vp = 0.5 * (R + iR.T) 
    Vm = 0.5 * (R - iR.T) 
    V0 = np.sqrt(0.5) * Wp @ (a-ap) 
    
    iVp = np.linalg.inv(Vp) 
    
    S = iVp 
    T = -iVp @ Vm 
    r = -iVp @ V0 
    
    Tp = Vm @ iVp 
    # Sp = (Vp - Vm @ iVp @ Vm)  = S^T
    rp = V0 - Vm @ iVp @ V0 
    
    
    return S, T, Tp, r, rp
    
def genRecOverlap(m,n,S,T,Tp,r,rp):
    """ 
    Calculate the overlap integral with a general recursive algorithm
    
    Parameters
    ----------
    m : (n,) array_like
        The state index of the bra function.
    n : (n,) array_like
        The state index of the ket function.
    S, T, Tp : (n,n) ndarray
        The recursion coefficients.
    r, rp : (n,) ndarray
        The recusrion coefficients.
        
    Returns
    -------
    float
        The overlap normalized to the vacuum integral, 
        :math:`\\langle m \\vert \\vert n \\rangle /\\langle 0 \\vert \\vert 0 \\rangle``
    
    Notes
    -----
    
    This routine uses direct recursive function calls and does not take advantage
    of lower-degree integrals, making it relatively inefficient. It it intended
    for debugging and verification purposes.
    
    See Also
    --------
    calcTwoCenterRecursion
    
    """
    
    ndim = len(m) 
    
    if sum(m) == 0:
        if sum(n) == 0: 
            return 1.0 
        else:
            # try to reduce n 
            for i in range(ndim):
                if n[i] > 0: 
                    
                    nnew = [ni for ni in n]
                    nnew[i] -= 1 
                    
                    val = rp[i] * genRecOverlap(m,nnew, S, T, Tp, r, rp) 
                    
                    for j in range(ndim):
                        if nnew[j] > 0 : 
                            nnew2 = [nj for nj in nnew]
                            nnew2[j] -= 1 
                            val += (Tp[i,j] * np.sqrt(nnew[j])) * genRecOverlap(m, nnew2, S, T, Tp, r, rp)
                    
                    for j in range(ndim):
                        if m[j] > 0 : 
                            mnew = [mj for mj in m]
                            mnew[j] -= 1 
                            val += (S[j,i] * np.sqrt(m[j])) * genRecOverlap(mnew, nnew, S, T,Tp,  r, rp) 
                                # Note Sp = S^T
                    val /= np.sqrt(n[i])
                    
                    return val 
        
    else:
        # try to reduce m 
        for i in range(ndim):
            if m[i] > 0:
                # reduce m_i 
                
                # three terms 
                mnew = [mi for mi in m] 
                mnew[i] -= 1 
                
                val = r[i] * genRecOverlap(mnew, n, S, T, Tp, r, rp) 
                
                for j in range(ndim):
                    if n[j] > 0: 
                        nnew = [nj for nj in n]
                        nnew[j] -= 1 
                        val += (S[i,j] * np.sqrt(n[j])) * genRecOverlap(mnew, nnew, S, T, Tp, r, rp)
                
                for j in range(ndim):
                    if mnew[j] > 0:
                        mnew2 = [mj for mj in mnew] 
                        mnew2[j] -= 1 
                        val += (T[i,j] * np.sqrt(mnew[j])) * genRecOverlap(mnew2, n, S, T,Tp, r, rp) 
                        
                val /= np.sqrt(m[i])
                
                return val 
            
    # Should not reach here! 
    raise RuntimeError("Entered recursively inaccesible return!")
    return 0.0 


def operator_degree0(v, Sigma):
    """
    Calculate the matrix representation of a constant 
    value.

    Parameters
    ----------
    v : (N,N) ndarray
        The constant value.
    Sigma : (N,nb,N,nb) ndarray
        The block overlap matrix

    Returns
    -------
    H : (N,nb,N,nb) 
        The block matrix represention 
    
    See Also
    --------
    calcSigma
        

    """
    
    N = v.shape[0] 
    
    v = np.reshape(v, (N,1,N,1))
    
    H = Sigma * v 
    
    return H 

def operator_degree1(da, db, Sigma, idxtab, dectab):
    """
    Calculate the matrix representation of a linear
    operator.
    
    Parameters
    ----------
    da : (N,N,n) ndarray
        The :math:`d_a` vector between each center 
    db : (N,N,n) ndarray
        The :math:`d_b` vector between each center 
    Sigma : (N,nb,N,nb) ndarray
        The block overlap matrix 
    idxtab : (nb,n) ndarray
        The excitation index table 
    dectab : (nb+1,n) ndarray
        The decrement look-up table 
        
    Returns
    -------
    H : (N,nb,N,nb)
        The block matrix representation 
    
    Notes
    -----
    
    Each two-center block has an operator defined by
    
    ..  math::
        
        \\langle m \\vert d_a^T a^\dagger + d_b^T \\beta \\vert n \\rangle 
    
    See Also
    --------
    calcSigma 
    buildDecrementTable 
    
    """
    
    N = da.shape[0] 
    ni = da.shape[2]
    nb = Sigma.shape[1] 
    
    H = np.zeros((N,nb,N,nb)) 
    
    for i in range(nb):
        m = idxtab[i]
        for j in range(nb):
            n = idxtab[j]
            
            # Compute <m|...|n>
            for k in range(ni):
                
                if m[k] > 0:
                    ip = dectab[i,k] # <m_k - 1| 
                    H[:,i,:,j] += np.sqrt(m[k]) * da[:,:,k] * Sigma[:,ip,:,j]  
                
                if n[k] > 0:
                    jp = dectab[j,k] # |n_k -1>
                    H[:,i,:,j] += np.sqrt(n[k]) * db[:,:,k] * Sigma[:,i,:,jp] 
    
    return H 

def operator_degree2( Daa, Dab, Dbb, Sigma, idxtab, dectab):
    """
    Calculate the matrix representation of a quadratic
    operator.
    
    Parameters
    ----------
    Daa : (N,N,n,n) ndarray
        The :math:`D_{aa}` matrix between each center 
    Dab : (N,N,n,n) ndarray
        The :math:`D_{ab}` matrix between each center
    Dbb : (N,N,n,n) ndarray
        The :math:`D_{bb}` matrix between each center
    Sigma : (N,nb,N,nb) ndarray
        The block overlap matrix 
    idxtab : (nb,n) ndarray
        The excitation index table 
    dectab : (nb+1,n) ndarray
        The decrement look-up table 
        
    Returns
    -------
    H : (N,nb,N,nb)
        The block matrix representation 
    
    Notes
    -----
    
    Each two-center block has an operator defined by
    
    ..  math::
        
        \\langle m \\vert a^{\\dagger T} D_{aa} a^\\dagger + \\beta^T D_{bb} \\beta  
        + a^{\\dagger T} D_{ab} \\beta  \\vert n \\rangle .
        
    
    
    See Also
    --------
    calcSigma 
    buildDecrementTable 
    
    """
    
    N,ni = Daa.shape[0], Daa.shape[2]
    nb = Sigma.shape[1] 
    
    H = np.zeros((N,nb,N,nb)) 
    
    for i in range(nb):
        m = idxtab[i]
        for j in range(nb):
            n = idxtab[j]
            
            # Compute <m|...|n>
            for k in range(ni):
                for l in range(ni):
                    
                    # D_aa 
                    if k == l:
                        if m[k] > 1: 
                            ip = dectab[dectab[i,k],l] 
                            H[:,i,:,j] += np.sqrt(m[k]*(m[k]-1)) * Daa[:,:,k,l] * Sigma[:,ip,:,j]
                    elif m[k] > 0 and m[l] > 0: 
                        ip = dectab[dectab[i,k],l] 
                        H[:,i,:,j] += np.sqrt(m[k] * m[l]) * Daa[:,:,k,l] * Sigma[:,ip,:,j]
                    
                    # D_bb 
                    if k == l: 
                        if n[k] > 1:
                            jp = dectab[dectab[j,k],l]
                            H[:,i,:,j] += np.sqrt(n[k]*(n[k]-1)) * Dbb[:,:,k,l] * Sigma[:,i,:,jp] 
                    elif n[k] > 0 and n[l] > 0:
                        jp = dectab[dectab[j,k],l] 
                        H[:,i,:,j] += np.sqrt(n[k] * n[l]) * Dbb[:,:,k,l] * Sigma[:,i,:,jp]
                    
                    # D_ab 
                    if m[k] > 0 and n[l] > 0 :
                        ip = dectab[i,k]
                        jp = dectab[j,l]
                        H[:,i,:,j] += np.sqrt(m[k]*n[l]) * Dab[:,:,k,l] * Sigma[:,ip,:,jp]
                    
    
    return H 

def operator_degree3( Daaa, Daab, Dabb, Dbbb, Sigma, idxtab, dectab):
    """
    Calculate the matrix representation of a cubic normal-ordered
    operator.
    
    Parameters
    ----------
    Daaa : (N,N,n,n,n) ndarray
        The :math:`D_{aaa}` matrix between each center 
    Daab : (N,N,n,n,n) ndarray
        The :math:`D_{aab}` matrix between each center 
    Dabb : (N,N,n,n,n) ndarray
        The :math:`D_{abb}` matrix between each center
    Dbbb : (N,N,n,n,n) ndarray
        The :math:`D_{bbb}` matrix between each center
    Sigma : (N,nb,N,nb) ndarray
        The block overlap matrix 
    idxtab : (nb,n) ndarray
        The excitation index table 
    dectab : (nb+1,n) ndarray
        The decrement look-up table 
        
    Returns
    -------
    H : (N,nb,N,nb)
        The block matrix representation 
    
    Notes
    -----
    
    Each two-center block has an operator defined by
    
    ..  math::
        
        \\sum_{ijk} (D_{aaa})_{ijk} \\alpha^\\dagger_i \\alpha^\\dagger_j \\alpha^\\dagger_k 
        + (D_{aab})_{ijk} \\alpha^\\dagger_i \\alpha^\\dagger_j \\beta_k 
        + (D_{abb})_{ijk} \\alpha^\\dagger_i \\beta_j \\beta_k 
        + (D_{bbb})_{ijk} \\beta_i \\beta_j \\beta_k 
        
    
    
    See Also
    --------
    calcSigma 
    buildDecrementTable 
    
    """
    
    N,ni = Daaa.shape[0], Daaa.shape[2]
    nb = Sigma.shape[1] 
    
    H = np.zeros((N,nb,N,nb)) 
    
    for i in range(nb):
        # m = idxtab[i]
        for j in range(nb):
            # n = idxtab[j]
            
            # Compute <m|...|n>
            raise NotImplementedError("degree 3 op not ready")
            
            for p in range(ni):
                for q in range(ni):
                    for r in range(ni):
                        
                        #
                        # (D_...)[p,q,r]
                        #
                        
                        # D_aaa 
                        ip, factor = _apply_annihilation_string(i, [p,q,r], idxtab, dectab)
                        jp = j 
                        if ip >= 0:
                            H[:,ip,:,jp] += factor * Daaa[:,:,p,q,r] * Sigma[:,ip,:,jp]
                        
                        # D_aab 
                        ip, factor1 = _apply_annihilation_string(i, [p,q], idxtab, dectab)
                        jp, factor2 = _apply_annihilation_string(j, [r], idxtab, dectab)
                        factor = factor1 * factor2 
                        if ip >=0 and jp >= 0:
                            H[:,ip,:,jp] += factor * Daab[:,:,p,q,r] * Sigma[:,ip,:,jp]
                            
                        # D_abb
                        ip, factor1 = _apply_annihilation_string(i, [p], idxtab, dectab)
                        jp, factor2 = _apply_annihilation_string(j, [q,r], idxtab, dectab)
                        factor = factor1 * factor2 
                        if ip >=0 and jp >= 0:
                            H[:,ip,:,jp] += factor * Dabb[:,:,p,q,r] * Sigma[:,ip,:,jp]
                            
                        # D_bbb
                        ip = i 
                        jp, factor = _apply_annihilation_string(j, [p,q,r], idxtab, dectab)
                        if jp >= 0:
                            H[:,ip,:,jp] += factor * Dbbb[:,:,p,q,r] * Sigma[:,ip,:,jp]
    
    
    return H 

def _apply_annihilation_string(i, modes, idxtab, dectab):
    """
    |n> = idxtab[i]
    
    apply annihilation operator for each entry in modes
    
    keep track of the sqrt[...] factors and the
    final state
    """
    
    # Initialize 
    ip = i 
    factor = 1.0 
    
    for p in modes:
        factor *= np.sqrt(idxtab[ip, p]) # sqrt[...] for this mode
        ip = dectab[ip, p]               # new state index 
        
    # if ip is -1, then the state was annihilated 
    # and factor will be meaningless (though it will always
    # return 0 if I understand my code correctly)
        
    return ip, factor 
    
    

#######################
# Hand-coded for small excitations 
#
def _overlap01(STTp,rrp):
    """
    (0,1) excitation block 
    """
    
    S,T,Tp = STTp 
    r,rp = rrp 
    n = len(r) 
    
    SS = np.zeros((1,n)) 
    
    SS[0,:] = rp 
    
    return SS 

def _overlap10(STTp,rrp):
    """
    (1,0) excitation block 
    """
    
    S,T,Tp = STTp 
    r,rp = rrp 
    n = len(r) 
    
    SS = np.zeros((n,1)) 
    
    SS[:,0] = r 
    
    return SS 

def _overlap11(STTp,rrp):
    """
    (1,1) block 
    """
    S,T,Tp = STTp 
    r,rp = rrp 
    # n = len(r) 
    
    SS = np.outer(r,rp) + S 
    
    return SS 

def _overlap02(STTp,rrp):
    """
    (0,2) block
    """
    S,T,Tp = STTp 
    r,rp = rrp 
    n = len(r) 
    
    n2 = (n*(n+1)) // 2
    SS = np.zeros((1,n2)) 
    
    idx = 0 
    for i in range(n):
        for j in range(i,n): 
            # <0 | 1i 1j>
            
            SS[0,idx] = rp[i] * rp[j] + Tp[i,j] 
            if i == j:
                SS[0,idx] *= np.sqrt(0.5) 
            
            idx += 1
    
    return SS 

def _overlap20(STTp,rrp):
    """
    (2,0) block
    """
    S,T,Tp = STTp 
    r,rp = rrp 
    n = len(r) 
    
    n2 = (n*(n+1)) // 2
    SS = np.zeros((n2,1)) 
    
    idx = 0 
    for i in range(n):
        for j in range(i,n): 
            # < 1i 1j | 0 >
            
            SS[idx,0] = r[i] * r[j] + T[i,j] 
            if i == j:
                SS[idx,0] *= np.sqrt(0.5) 
            
            idx += 1
    
    return SS 

def _overlap12(STTp,rrp):
    """
    (1,2) block
    """
    S,T,Tp = STTp 
    r,rp = rrp 
    n = len(r) 
    
    n2 = (n*(n+1)) // 2
    SS = np.zeros((n,n2)) 

    for i in range(n):
        idx = 0 
        for j in range(n):
            for k in range(j,n):
                
                SS[i,idx] = r[i]*(rp[j]*rp[k] + Tp[j,k]) + S[i,j]*rp[k] + S[i,k]*rp[j]
                if j == k:
                    SS[i,idx] *= np.sqrt(0.5) 
                    
                idx += 1
    
    return SS 

def _overlap21(STTp,rrp):
    """
    (2,1) block
    """
    S,T,Tp = STTp 
    r,rp = rrp 
    n = len(r) 
    
    n2 = (n*(n+1)) // 2
    SS = np.zeros((n2,n)) 

    idx = 0 
    for i in range(n):
        for j in range(i,n):
            for k in range(n):
                
                SS[idx,k] = rp[k] * (r[i] * r[j] + T[i,j]) + S[i,k] * r[j] + S[j,k] * r[i]
                if i == j:
                    SS[idx,k] *= np.sqrt(0.5) 

            idx += 1
    
    return SS 

def _overlap22(STTp,rrp):
    """
    (2,2) block
    """
    S,T,Tp = STTp 
    r,rp = rrp 
    n = len(r) 
    
    n2 = (n*(n+1)) // 2
    SS = np.zeros((n2,n2)) 
    
    idx1 = 0 
    for i in range(n):
        for j in range(i,n):
            
            idx2 = 0 
            for k in range(n):
                for l in range(k,n):
                    
                    SS[idx1,idx2] = r[i] * r[j] * rp[k] * rp[l] +\
                        r[i]*r[j]*Tp[k,l] + T[i,j]*rp[k]*rp[l] + T[i,j]*Tp[k,l] +\
                        r[i]*S[j,k]*rp[l] + r[i]*S[j,l]*rp[k] + r[j]*S[i,k]*rp[l] + r[j]*S[i,l]*rp[k] +\
                        S[i,k]*S[j,l] + S[i,l]*S[j,k]
                        
                    if i == j:
                        SS[idx1,idx2] *= np.sqrt(0.5)
                    if k == l:
                        SS[idx1,idx2] *= np.sqrt(0.5) 
                        
                    idx2 += 1 
            
            idx1 += 1
            
    return SS

def buildDecrementTable(n,nex):
    """
    Construct a multi-index decrement look-up table.

    Parameters
    ----------
    n : integer
        The number of modes
    nex : integer
        The maximum excitation number.

    Returns
    -------
    dectab : (nb+1,n) ndarray
        The look-up table. `dectab`[i,j] is the index position 
        of the multi-index that results from decrementing mode `j`
        from the multi-index `i`. Entries of ``-1`` indicate an 
        invalid decrement (i.e. decrementing 0).
        `dectab[-1]`, i.e. the last row, will always be
        ``[-1, -1, ..., -1]``, allowing for recursive indexing 
        into `dectab` with arbitary depth.

    """
    
    tab = adf.idxtab(nex,n) # The multi-index table 
    nb = len(tab) 
    dectab = np.empty((nb+1,n), dtype = np.int32) 
    nck = adf.ncktab(n+nex)

    for i in range(nb):
        m = tab[i] # This multi-index 
        
        for j in range(n):
            # Try decreasing m[i] 
            if m[j] == 0:
                # Invalid, store -1
                dectab[i,j] = -1 
            else: 
                m[j] -= 1 
                new_idx = adf.idxpos(m, nck) 
                m[j] += 1 
                dectab[i,j] = new_idx 
    
    dectab[-1,:] = -1 
                
    return dectab 

def block2matrix(M):
    """
    Reshape a pair-center block matrix to 2d array

    Parameters
    ----------
    M : (...,N,nb,N,nb) ndarray
        The block matrix

    Returns
    -------
    (...,N*nb,N*nb) ndarray
        The reshaped array 

    """
    base_shape = M.shape[:-4]
    N,nb = M.shape[-2], M.shape[-1]
    return np.reshape(M, base_shape + (N*nb, N*nb), )

def calcDHOBasisParams(qcenters, cs, V, masses, method = 'adiabatic_two_well',
                       fscale = 0.4, disp = False):
    """
    
    Calculate parameters for a Distributed
    HO basis set.
    
    Parameters
    ----------
    qcenters : (nq,N) array_like
        The basis set centers.
    cs : CoordSys
        The coordinate system
    V : DFun
        The potential energy surface 
    masses : array_like
        The masses.
    method : {'adiabatic_two_well'}
        The construction method. See Notes
    fscale : float, optional
        The tangent width scaling factor. The default is 0.4
    disp : bool, optional
        If True, print and plot basis information.

    Returns
    -------
    a : (N,nq) ndarray
        The basis centers
    W : (N,nq,nq) ndarray
        The basis normal-mode tranformation arrays
    
    
    Notes
    -----
    
    ``'adiabatic_two_well'`` : The path in `qcenters` is assumed to connect 
    two local minima. The full set of normal modes at the first and last 
    center are used as-is. At each intermediate point, the `nq`-1 normal modes
    orthogonal to the path-tangent (which is approximated by finite differences)
    are used as well as the path tangent.  The displacement vector of the
    tangent mode at center :math:`q_i`, 
    which defines the width of the basis functions along that
    dimension, is equal to :math:`f_\\mathrm{scale} \\times (q_{i+1} - q_{i-1})`,
    where :math:`f_\\mathrm{scale}` is an adjustable input parameter.

    """
    
    
    qcenters = np.array(qcenters)
    nq,N = qcenters.shape 
    
    if method == 'adiabatic_two_well':
    
        ##############################
        # Calculate the projected vibrations along the 
        # qcenter path
        #
        # Define the inverse vibrational metric
        GFun = n2.pes.rxnpath.InverseMetric(cs, masses = masses)
        fproj = np.zeros((nq,N-2)) # The tangent vector 
        for i in range(N-2):
            fproj[:,i] = qcenters[:,i+2] - qcenters[:,i]
        omega, T, Tl = n2.pes.rxnpath.pathvib_nonstationary(qcenters[:,1:-1], V, GFun, fproj = fproj)
        
        if disp:
            plt.figure() 
            x = np.arange(N-2) + 1 
            plt.plot(x, omega.T,'m.')
        
            
        #################
        # Calculate DG parameters 
        ti = []
        Wi = [] 
        
        ########################
        # First edge point
        # 
        q0 = qcenters[:,0]
        # Calculate the normal modes of the first well
        ti.append(q0)
        w0,nc = n2.pes.curvVib(q0, V, cs, masses)
        iT = np.linalg.inv(nc.T)
        Wi.append(iT)
        
        if disp:
            plt.plot(0*w0, w0, 'kx')
        #
        ########################
        
        ########################
        # Interior points
        for i in range(N-2):
            ti.append(qcenters[:,i+1])
            Ti = T[:,:,i]
            fi = fproj[:,i].reshape((-1,1))
            Ti = np.concatenate([Ti,fi*fscale], axis = 1) # the final column is the path tangent
            iT = np.linalg.inv(Ti) # the final row corresponds to the path tangent 
            Wi.append(iT)
        #
        ########################
        
        ########################
        # Second edge point
        q1 = qcenters[:,-1]
        ti.append(q1)
        w1,nc = n2.pes.curvVib(q1, V, cs, masses)
        iT = np.linalg.inv(nc.T)
        Wi.append(iT) 
        
        if disp:
            plt.plot(0*w1 + (N-1), w1, 'kx')
        #
        #######################
    
    else:
        raise ValueError('Invalid method.')
    
    a = np.array(ti)
    W = np.array(Wi)
    
    return a, W

def quadraticVib(DHO, V, cs, masses, printlevel = 1,
                         hbar = None):
    """
    Calculate tunneling splitting with a local quadratic approximation.

    Parameters
    ----------
    DHO : DistHOBasis
        A distributed harmonic oscillator basis set.
    V : DFun
        The potential energy surface.
    cs : CoordSys
        The coordinate system
    masses : array_like
        The masses.
    printlevel : integer, optional
        The print level. The default is 1.
    hbar : float, optional
        The value of :math:`\\hbar`. If None (default), standard units are used.

    Returns
    -------
    w : (N,) ndarray
        The vibrational eigenenergies.

    """


    nq = DHO.n  # The number of coordinates
    N = DHO.N   # The number of centers 
    
    if printlevel >= 1:
        print("--------------------------------------")
        print(" Quadratic distributed HO calculation ")
        print("")
        print(" There are                            ")
        print(f"         N = {N:d} centers and       ")
        print(f" N*(N+1)/2 = {(N*(N+1))//2:d} pair-centers")
        print("--------------------------------------")
        
    ###############################################
    # Calculate the PES quadratic force-fields.
    if printlevel >= 1:
        print("Calculating quadratic PES expansions...",end = "")
    start = time.perf_counter()
    dV = V.f(DHO.c.T, deriv = 2)[:,0]
    nck = adf.ncktab(nq+3) 
    # Expand the derivative arrays
    partials = adf.mvexpand(dV, 2, nq, nck)
    # 
    stop = time.perf_counter()
    if printlevel >= 1:
        print(f"done \n ({stop-start:.3f} s)")
    
    # The value, gradient, and hessian
    vval = partials[0] 
    vjac = partials[1] 
    vhes = partials[2] 

    #################################################
    # Calculate kinetic energy operator
    #
    if hbar is None:
        hbar = n2.constants.hbar
    
    
    if printlevel >= 1:
        print("")
        print("Calculating KEO expansions...",end = "")
    start = time.perf_counter()
    dGV, _, _, _, _ = cs.Q2GUV(DHO.c.T, masses, 0, hbar = hbar)
    stop = time.perf_counter()
    if printlevel >= 1:
        print(f"done \n ({stop-start:.3f} s)")
        
    G = dGV[0] # G * hbar**2/2

    nex = 0
    if printlevel >= 1:
        print("")
        print("Calculating overlap integrals...",end = "")
    start = time.perf_counter()
    Sigma = DHO.calcSigma(nex) # The overlap integrals
    N,nb = Sigma.shape[0:2]
    stop = time.perf_counter()
    if printlevel >= 1:
        print(f"done \n ({stop-start:.3f} s)")

    idxtab = adf.idxtab(nex, nq)
    dectab = buildDecrementTable(nq, nex)

    #
    if printlevel >= 1:
        print("")
        print("Calculating normal-ordered operators...",end = "")
    start = time.perf_counter()
    
    # Calculate the normal-ordered expressions of the PES
    d0, da, db, Daa, Dab, Dbb = DHO.normal_order_coordinate2(vval, vjac.T, vhes.T)
    
    # Calculate the PES integrals up to quadratic terms
    VHO = np.zeros_like(Sigma)
    VHO += operator_degree0(d0, Sigma)
    VHO += operator_degree1(da, db, Sigma, idxtab, dectab)
    VHO += operator_degree2(Daa, Dab, Dbb, Sigma, idxtab, dectab)

    
    # Calculate the normal-ordered expression of the KEO
    d0,da,db,Daa,Dab,Dbb = DHO.normal_order_deriv2(G.T) 
    THO = np.zeros_like(Sigma) 
    THO += operator_degree0(d0, Sigma)
    THO += operator_degree1(da, db, Sigma, idxtab, dectab)
    THO += operator_degree2(Daa, Dab, Dbb, Sigma, idxtab, dectab)
    THO *= -1
    
    stop = time.perf_counter()
    if printlevel >= 1:
        print(f"done \n ({stop-start:.3f} s)")
        
    # Reshape matrices from block form to matrix form
    Sigma = block2matrix(Sigma)
    VHO = block2matrix(VHO)
    THO = block2matrix(THO)

    # Diagonalize the Hamlitonian
    
    if printlevel >= 1:
        print("")
        print(f"Diagonalizing Hamiltonian (N * nb = {N*nb:d}) ...",end = "")
    start = time.perf_counter()
    HHO = THO + VHO
    wHO, UHO = eigh(HHO, Sigma)
    stop = time.perf_counter()
    if printlevel >= 1:
        print(f"done \n ({stop-start:.3f} s)")
    
    e0, e1 = wHO[0:2]
    
    if printlevel >= 1:
        print("")
        print("The DHO ZPE energy = ", end = "")
        print(e0)
        print("The DHO splitting =  ", end = "")
        print(e1 - e0)

    return wHO 

def cubicRoVib(DHO, V, cs, masses, nex = 1, 
               printlevel = 1, hbar = None,
               prune_index = -1):
    """
    Calculate tunneling splitting and rotational
    contsants with a local cubic approximation.

    Parameters
    ----------
    DHO : DistHOBasis
        A distributed harmonic oscillator basis set.
    V : DFun
        The potential energy surface.
    cs : CoordSys
        The coordinate system
    masses : array_like
        The masses.
    nex : integer, optional
        The maximum vibrational excitation. The default is 1. 
    printlevel : integer, optional
        The print level. The default is 1.
    hbar : float, optional
        The value of :math:`\\hbar`. If None (default), standard units are used.
    prune_index : integer, optional
        The vibrational index to prune for internal-center excitations. The default is -1 (the last).
        If None, no excitation functions will be pruned.

    Returns
    -------
    w : (N,) ndarray
        The vibrational eigenenergies.
    sigma : (3,3,2,2) ndarray
        The effective quadratic angular momentum coefficients.
    tau : (3,3,3,3,2,2) ndarray
        The effective quartic angular momentum coefficients.

    """


    nq = DHO.n  # The number of coordinates
    N = DHO.N   # The number of centers 
    
    if printlevel >= 1:
        print("--------------------------------------")
        print(" Cubic distributed HO calculation     ")
        print("")
        print(" There are                            ")
        print(f"         N = {N:d} centers and       ")
        print(f" N*(N+1)/2 = {(N*(N+1))//2:d} pair-centers")
        print(" The excitation count is ")
        print(f" nex = {nex:d}")
        print("--------------------------------------")
        
    ###############################################
    # Calculate the PES cubic force-fields.
    if printlevel >= 1:
        print("Calculating cubic PES expansions...",end = "")
    start = time.perf_counter()
    dV = V.f(DHO.c.T, deriv = 3)[:,0]
    nck = adf.ncktab(nq+3) 
    # Expand the derivative arrays
    partials = adf.mvexpand(dV, 3, nq, nck)
    # 
    stop = time.perf_counter()
    if printlevel >= 1:
        print(f"done \n ({stop-start:.3f} s)")
    
    # The value, gradient, and hessian
    vval = partials[0] 
    vjac = partials[1] 
    vhes = partials[2] 
    phi3 = partials[3] 

    #################################################
    # Calculate kinetic energy operator
    #
    if hbar is None:
        hbar = n2.constants.hbar
    
    
    if printlevel >= 1:
        print("")
        print("Calculating quadratic KEO expansions...",end = "")
    start = time.perf_counter()
    dGV, dGrv, dGr, dU, dVT = cs.Q2GUV(DHO.c.T, masses, 2, hbar = hbar)
    stop = time.perf_counter()
    if printlevel >= 1:
        print(f"done \n ({stop-start:.3f} s)")
        
    G,G_jac,_ = adf.mvexpand(dGV, 2, nq, nck) # G * (hbar**2/2)
    G_jac = np.moveaxis(G_jac, 0, 2)
    
    #
    # (Don't bother with pseudo U terms)
    # Uval, Ujac = adf.mvexpand(dU, 1, nq, nck) 
    # trU = np.zeros((N,N)) # trace of Ujac
    # for i in range(nq):
    #     trU += Ujac[i,i] 

    
    gr, j_gr, h_gr = adf.mvexpand(dGr, 2, nq, nck) # G_rr * (hbar**2/2)
    j_gr = np.moveaxis(j_gr, [0], [2])
    h_gr = np.moveaxis(h_gr, [0,1], [2,3])

    grv, j_grv, _ = adf.mvexpand(dGrv, 2, nq, nck) # G_rv * (hbar**2/2)
    j_grv = np.moveaxis(j_grv, [0], [2])


    if printlevel >= 1:
        print("")
        print("Calculating overlap integrals...",end = "")
    start = time.perf_counter()
    Sigma = DHO.calcSigma(nex) # The overlap integrals
    N,nb = Sigma.shape[0:2]
    stop = time.perf_counter()
    if printlevel >= 1:
        print(f"done \n ({stop-start:.3f} s)")

    idxtab = adf.idxtab(nex, nq)
    dectab = buildDecrementTable(nq, nex)

    #
    if printlevel >= 1:
        print("")
        print("Calculating normal-ordered operators...",end = "")
    start = time.perf_counter()
    
    # Calculate the normal-ordered expressions of the PES
    d0, da, db, Daa, Dab, Dbb = DHO.normal_order_coordinate2(vval, vjac.T, vhes.T)
    
    # Calculate the PES integrals up to quadratic terms
    VHO = np.zeros_like(Sigma)
    VHO += operator_degree0(d0, Sigma)
    VHO += operator_degree1(da, db, Sigma, idxtab, dectab)
    VHO += operator_degree2(Daa, Dab, Dbb, Sigma, idxtab, dectab)

    # Normal-order the cubic terms
    da,db,Daaa,Daab,Dabb,Dbbb = DHO.normal_order_phi3(phi3.T)
    V3HO = np.zeros_like(Sigma)
    V3HO += operator_degree1(da, db, Sigma, idxtab, dectab) 
    # Ignore the rank-3 contributions to V3 
    
    
    # Calculate the normal-ordered expression of the KEO
    d0,da,db,Daa,Dab,Dbb = DHO.normal_order_deriv2(G.T) 
    THO = np.zeros_like(Sigma) 
    THO += operator_degree0(d0, Sigma)
    THO += operator_degree1(da, db, Sigma, idxtab, dectab)
    THO += operator_degree2(Daa, Dab, Dbb, Sigma, idxtab, dectab)
    THO *= -1
    
    # The normal-order the G-jacobian terms
    d0,da,db,Daa,Dab,Dbb,Daaa,Daab,Dabb,Dbbb = DHO.normal_order_derivcoord21(G_jac.T)
    T3HO = np.zeros_like(Sigma) 
    T3HO += operator_degree0(d0, Sigma) 
    T3HO += operator_degree1(da, db, Sigma, idxtab, dectab) 
    T3HO += operator_degree2(Daa, Dab, Dbb, Sigma, idxtab, dectab) 
    T3HO *= -1
    # ignore rank-3 contributions to T3 
    
    # (Don't bother with pseudo U terms)
    # Upseudo = dh.operator_degree0(-trU, Sigma)
    
    # Rotational terms
    GrrHO = np.zeros((3,3,N,nb,N,nb))
    for a in range(3):
        for b in range(3):
            # G_ab 
            d0,da,db,Daa,Dab,Dbb = \
                DHO.normal_order_coordinate2(gr[a,b],j_gr[a,b].T,h_gr[a,b].T)
            GrrHO[a,b] += operator_degree0(d0, Sigma)
            GrrHO[a,b] += operator_degree1(da,db,Sigma,idxtab,dectab)
            GrrHO[a,b] += operator_degree2(Daa,Dab,Dbb,Sigma, idxtab, dectab)

    # Rovibrational terms
    GrvHO = np.zeros((3,N,nb,N,nb)) 
    for a in range(3):
        # G_ak
        d0,da,db,Daa,Dab,Dbb = \
            DHO.normal_order_derivcoord11(grv[a].T, j_grv[a].T) 
        GrvHO[a] += operator_degree0(d0, Sigma)
        GrvHO[a] += operator_degree1(da,db,Sigma,idxtab,dectab)
        GrvHO[a] += operator_degree2(Daa,Dab,Dbb,Sigma, idxtab, dectab)
        
        
    stop = time.perf_counter()
    if printlevel >= 1:
        print(f"done \n ({stop-start:.3f} s)")
        
    # Reshape matrices from block form to matrix form
    Sigma = block2matrix(Sigma)
    VHO = block2matrix(VHO)
    THO = block2matrix(THO)
    
    V3HO = block2matrix(V3HO)
    T3HO = block2matrix(T3HO)
    GrrHO = block2matrix(GrrHO)
    GrvHO = block2matrix(GrvHO) 

    ###############################
    # Prune basis 
    if prune_index is None:
        # No pruning
        if printlevel >= 1:
            print("The basis will not be pruned.")
    else:
        if printlevel >= 1:
            print(f"The basis will be pruned in mode index {prune_index:d}.")
        include = np.ones((N,nb), dtype = bool) 
        for i in range(N):
            for j in range(nb):
                if i > 0 and i < N-1:
                    # interior point 
                    if idxtab[j,prune_index] > 0:
                        # Excitation in path tangent 
                        include[i,j] = False 
        include = np.reshape(include, (-1,)) == True
        ix,iy = np.ix_(include, include)
        Sigma = Sigma[ix,iy] 
        VHO = VHO[ix,iy]
        V3HO = V3HO[ix,iy]
        THO = THO[ix,iy]
        T3HO = T3HO[ix,iy]
        GrrHO = GrrHO[:,:,ix,iy]
        GrvHO = GrvHO[:,ix,iy]
    #
    ##############################

    # Diagonalize the Hamlitonian
    
    if printlevel >= 1:
        print("")
        NH = len(VHO)
        print(f"Diagonalizing zeroth-order Hamiltonian (N(total) = {NH:d}) ...",end = "")
    start = time.perf_counter()
    HHO = THO + VHO
    wHO, UHO = eigh(HHO, Sigma)
    stop = time.perf_counter()
    if printlevel >= 1:
        print(f"done \n ({stop-start:.3f} s)")
    
    e0, e1 = wHO[0:2]
    
    if printlevel >= 1:
        print("")
        print("The DHO[0] ZPE energy = ", end = "")
        print(e0)
        print("The DHO[0] splitting =  ", end = "")
        print(e1 - e0)
        
    ################################
    # Calculate the effective rotational Hamiltonians
    #
    if printlevel >= 1:
        print("Transforming matrix elements ... ", end = "")
    start = time.perf_counter()
    
    Grr_t = np.zeros_like(GrrHO)
    Grv_t = np.zeros_like(GrvHO)
    
    for a in range(3):
        for b in range(3):
            np.copyto(Grr_t[a,b], -UHO.T @ GrrHO[a,b] @ UHO)
        np.copyto(Grv_t[a], -UHO.T @ GrvHO[a] @ UHO)
        
    V3_t = UHO.T @ V3HO @ UHO 
    T3_t = UHO.T @ T3HO @ UHO 
    H3_t = V3_t + T3_t 
    
    if printlevel >= 1:
        print(f"done \n ({stop-start:.3f} s)")

    Delta = np.reshape(wHO[:2], (-1,1)) - np.reshape(wHO[2:], (1,-1))

    # c = 29979.2458
 
    H2_v0_r = np.zeros((3,3,2,2))  # Vib[0] x Rot[0,1,2] (i.e. Harmonic)
    H2_rv_rv = np.zeros((3,3,2,2)) # Rovib[1] x Rovib[1] (i.e. Coriolis)
    H2_v1_r = np.zeros((3,3,2,2))  # Vib[1] x Rot[0,1,2] (i.e. Anharmonic)
    
    H4 = np.zeros((3,3,3,3,2,2))
    
    for a in range(3):
        for b in range(3):
            
            H2_v0_r[a,b] += Grr_t[a,b,:2,:2] 
            
            H2_rv_rv[a,b] += 0.5 * (Grv_t[a,:2,2:] / Delta) @ Grv_t[b,2:,:2] 
            H2_rv_rv[a,b] += 0.5 * Grv_t[a,:2,2:] @ (Grv_t[b,2:,:2] / Delta.T)
            
            H2_v1_r[a,b] += 0.5 * (Grr_t[a,b,:2,2:] / Delta) @ H3_t[2:,:2]
            H2_v1_r[a,b] += 0.5 * Grr_t[a,b,:2,2:] @ (H3_t[2:,:2] / Delta.T) 
            H2_v1_r[a,b] += 0.5 * (H3_t[:2,2:] / Delta) @ Grr_t[a,b,2:,:2]
            H2_v1_r[a,b] += 0.5 * H3_t[:2,2:] @ (Grr_t[a,b,2:,:2] / Delta.T)
            
            for c in range(3):
                for d in range(3):
                    H4[a,b,c,d] += 0.5 * (Grr_t[a,b,:2,2:] / Delta) @ Grr_t[c,d,2:,:2]
                    H4[a,b,c,d] += 0.5 * Grr_t[a,b,:2,2:] @ (Grr_t[c,d,2:,:2] / Delta.T)
                    
    
    
    H2 = H2_v0_r + H2_rv_rv + H2_v1_r   # Total H_eff
    
    if printlevel >= 1:
        
        for scale in [-1 , -29979.2458]:
            
            print("="*90)
            print(f"Scale = {scale:f}")
            
            print("")
            print("V[0] x R[0,1,2] (""Harmonic"")")
            _printHrot(scale * H2_v0_r)
            
            print("")
            print("RV[1] x RV[1] (""Coriolis"")" )
            _printHrot(scale * H2_rv_rv)
            
            print("")
            print("V[1] x R[0,1,2] (""Anharmonic"")")
            _printHrot(scale * H2_v1_r)
            
            print("")
            print("Total")
            _printHrot(scale * H2)
            
            
    sigma = -2 * H2 
    tau = 4 * H4 
    
    return wHO, sigma, tau 

def _printHrot(heff):
    
    ndash = 84
    
    print("-"*ndash)
    for i in range(2):
        for a in range(3):
            for j in range(2):
                for b in range(3):
                    
                    print(f"{heff[a,b,i,j]:12.4f} ", end = "")
               
                print(" | ", end = "") # Space between vib block
            
            print("") # New line after rotation index
            
        print("-"*ndash) # Extra space between vib block
                
    return 
            
            
                        