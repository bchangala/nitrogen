#cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True, language_level=3
# 
#

cimport nitrogen.autodiff.cyad.cyad_core as cyad

from libc.stdlib cimport malloc, free 

def pes(double [:,:,:] dX, double [:,:,:] dV, int order, int [:,:] table):

    """
    Evaluate the PES

    Parameters
    ----------
    double [nd,3*N,n] dX
        The 3*N Cartesian coordinates at `n` geometries
        
    double [nd,1,n] dV
        The energy output.
        
    int order 
        Derivative order 
        
    int [3,tablesize] table
        Product table 
        
    """
    
    cdef:
        size_t i,j,k,l,m
        size_t idx_jk, idx_kl, idx_jl 
        
        size_t nd = dX.shape[0] # the number of derivatives
        int N = dX.shape[1] // 3 # The number of atoms
        size_t n = dX.shape[2] # the number of geometries
        
        int nrho = ( N * (N-1) ) // 2 # The number of pairs
        
        double **dx = cyad.malloc2d(3*N, nd)  # Cartesian geometries 
        double **rho = cyad.malloc2d(nrho, nd) # Pair rho values (r_ij - re) / re
        
        size_t tablesize = table.shape[1] 
        size_t *idxZ = <size_t *>malloc(tablesize * sizeof(size_t))
        size_t *idxX = <size_t *>malloc(tablesize * sizeof(size_t))
        size_t *idxY = <size_t *>malloc(tablesize * sizeof(size_t))
        
        cdef cyad.adtab tab = cyad.adtab(order, nd, tablesize, idxZ, idxX, idxY)
        
        size_t tempsize = 16
        double **temp = cyad.malloc2d(tempsize, nd) # workspace 
        double *Ftemp = cyad.malloc1d(order + 1)    # workspace 
        
        # Force field parameters
        # MM^Vib
        #
        double D = 50796.795 # cm^-1 
        double re = 1.313 # Angstroms
        double a2 = 7.428 # dimensionless 
        double a3 = 8.072 
        
        double c0 = 7.788
        double c1 = 3.917 
        double c2 = -17.503
        double c3 = -51.427 # sign??
        double c4 = 99.263
        double c5 = -39.772 # sign??
        double c6 = 70.505
        double c7 = 73.262
        double c8 = 3.831
        double c9 = 65.696
        double c10 = -85.307 # sign??
        
        # Note: there are some sign differences with the 
        # above parameters in the thesis of the first author
        # of the main reference...
        
        # # Original MM^Eggen
        # #
        # double D = 50796.795 # cm^-1 
        # double re = 1.507 # Typo in the Tailor et al paper. Original Eggen et al has 1.507
        # double a2 = 8.200
        # double a3 = 8.200
        
        # double c0 = 8.087
        # double c1 = -13.334 
        # double c2 = 26.882
        # double c3 = -51.646
        # double c4 = 12.164
        # double c5 = 51.629
        # double c6 = 25.697
        # double c7 = -5.964
        # double c8 = -7.306
        # double c9 = 2.208
        # double c10 = 13.707
        
    # prepare table 
    for i in range(tablesize):
        idxZ[i] = table[0,i]
        idxX[i] = table[1,i]
        idxY[i] = table[2,i]
    
    
    for i in range(n):
        # For each geometry 
        
        # Get the inputs
        for j in range(3*N):
            for k in range(nd):
                dx[j][k] = dX[k,j,i] 
        
        
        # Calculate potential
        #
        # Initialize derivative array 
        for k in range(nd):
            dV[k,0,i] = 0.0 
            
            
        # Calculate rho values 
        calc_rho(dx, rho, re, <size_t> N, Ftemp, temp, &tab)
        
        ###################
        # Calculate the 2-body contribution 
        for j in range(nrho):
            calc_2body(temp[0], a2, D, rho[j], Ftemp, temp+1, &tab) 
            
            for k in range(nd):
                dV[k,0,i] += temp[0][k]
        #
        ###################
        
        ###################
        # Calculate the 3-body contributions
        for l in range(N):
            for k in range(l):
                idx_kl = (l*(l-1)) // 2 + k
                for j in range(k):
                    # j < k < l
                    idx_jk = (k*(k-1)) // 2 + j
                    idx_jl = (l*(l-1)) // 2 + j
                    
                    calc_3body(temp[0], c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10,
                               a3, D, 
                               rho[idx_jk], rho[idx_kl], rho[idx_jl],
                               Ftemp, temp+1, &tab) 
                    
                    for m in range(nd):
                        dV[m,0,i] += temp[0][m]
        
    cyad.free2d(dx, 3*N)
    cyad.free2d(rho, nrho) 
    free(idxZ)
    free(idxY)
    free(idxX)
    cyad.free2d(temp, tempsize)
    cyad.free1d(Ftemp) 
    
    return 


cdef void calc_rho(double **dx, double **rho, double re, size_t natoms, double *F, double **temp,  cyad.adtab *t):
    # Input
    #   x in Angstroms
    #
    # Output
    #   rho dimensionless
    
    # F [k+1] temp space
    # temp [7][nd]
    # 
    
    cdef int i,j,idx
    
    idx = 0
    for i in range(natoms): 
        for j in range(i): # j < i 
            cyad.sub(temp[0], dx[3*i+0], dx[3*j+0], t) # delta x
            cyad.sub(temp[1], dx[3*i+1], dx[3*j+1], t) # delta y 
            cyad.sub(temp[2], dx[3*i+2], dx[3*j+2], t) # delta z
            
            cyad.mul(temp[3], temp[0], temp[0], t) # dx * dx 
            cyad.mulacc(temp[3], temp[1], temp[1], t) # dy * dy 
            cyad.mulacc(temp[3], temp[2], temp[2], t) # dz * dz 
            # r2 = dx*dx + dy*dy + dz*dz 
            
            # rho <-- sqrt(r**2)
            cyad.sqrt(rho[idx], temp[3], F, temp + 4,  t)
            
            # calculate rho = (r - re) / re
            rho[idx][0] -= re 
            cyad.smul(rho[idx], 1.0 / re, rho[idx], t)
            
            idx += 1 
    return 

cdef void calc_2body(double *v, double a2, double D, double *rho,
                      double *F, double **temp, cyad.adtab *t):
    # Calculate 2-body function 
    # 
    # V(2) = +D * (-1 - a2 * rho_ij) * exp(-a2 * rho_ij)
    # 
    # F ... (k+1) work
    # temp ... (4,nd) work
    #
    
    # temp[0] <-- -a2 * rho 
    cyad.smul(temp[0], -a2, rho, t)
    
    # v <-- exp(temp[0])
    cyad.exp(v, temp[0], F, temp + 1, t)
    
    # temp[0] = -1 -a2*rho_ij
    temp[0][0] -= 1.0 
    
    # temp[1] = (-1 - a2*rho_ij) * exp(...)
    cyad.mul(temp[1], temp[0], v, t)
    
    # multiply by +D (scalar)
    cyad.smul(v, D, temp[1], t)
    
    # result is in `v` 
    
    return

cdef void calc_3body(double *v, double c0, double c1, double c2, 
                     double c3, double c4, double c5, double c6, 
                     double c7, double c8, double c9, double c10,
                     double a3, double D, 
                     double *rho_ij, double *rho_jk, double *rho_ik,
                     double *F, double **temp, cyad.adtab *t):
    #
    # Calculate 3-body function 
    #
    # temp ... 14
    #
    cdef: 
        double *Q1 = temp[0] 
        double *Q2 = temp[1]
        double *Q3 = temp[2] 
        
        double *P = temp[4] 
        double *Q1Q1 = temp[5] 
        double *Q2Q2_Q3Q3 = temp[6] 
        double *dum = temp[7] 
        
        double *Q2Q2 = temp[8]
        double *Q3Q3 = temp[9] 
        
        double *Qcub = temp[10] 
        
        double irt3 = 0.57735026918962576450914
        double irt2 = 0.70710678118654752440084
        double irt6 = 0.40824829046386301636621
    


    # Q1 <-- (ij + ik + jk) / rt(3)
    cyad.add(Q1, rho_ij, rho_ik, t)
    cyad.add(Q1, Q1, rho_jk, t)
    cyad.smul(Q1, irt3, Q1, t) 
    
    # Q2 <-- (ik - jk) / rt[2]
    cyad.sub(Q2, rho_ik, rho_jk, t) 
    cyad.smul(Q2, irt2, Q2, t) 
    
    # Q3 <-- (2*ij - ik - jk) / rt[6]
    cyad.smul(Q3, 2.0, rho_ij, t) 
    cyad.sub(Q3, Q3, rho_ik, t)
    cyad.sub(Q3, Q3, rho_jk, t) 
    cyad.smul(Q3, irt6, Q3, t) 
    


    # c0 and c1
    cyad.smul(P, c1, Q1, t) 
    P[0] += c0  
    
    
    # c2 
    cyad.mul(Q1Q1, Q1, Q1, t) 
    cyad.smulacc(P, c2, Q1Q1, t) 
    
    # c3 
    cyad.mul(Q2Q2, Q2, Q2, t) 
    cyad.mul(Q3Q3, Q3, Q3, t)
    cyad.add(Q2Q2_Q3Q3, Q2Q2, Q3Q3, t)
    cyad.smulacc(P, c3, Q2Q2_Q3Q3, t) 
    
    # c4 and c7
    cyad.mul(dum, Q1, Q1Q1, t) 
    cyad.smulacc(P, c4, dum, t) 
    cyad.mul(dum, Q1Q1, Q1Q1, t)
    cyad.smulacc(P, c7, dum, t) 
    
    # c5
    cyad.mul(dum, Q1, Q2Q2_Q3Q3, t)
    cyad.smulacc(P, c5, dum, t) 
    
    # c6 and c10 
    cyad.smul(dum, -3.0, Q2Q2, t)
    cyad.mul(Qcub, Q3, dum, t) 
    cyad.mulacc(Qcub, Q3, Q3Q3, t)
    cyad.smulacc(P, c6, Qcub, t) 
    
    cyad.mul(dum, Q1, Qcub, t)
    cyad.smulacc(P, c10, dum, t) 
    
    # c8
    cyad.mul(dum, Q1Q1, Q2Q2_Q3Q3, t) 
    cyad.smulacc(P, c8, dum, t) 
    
    # c9
    cyad.mul(dum, Q2Q2_Q3Q3, Q2Q2_Q3Q3, t)
    cyad.smulacc(P, c9, dum, t) 
    
    # P is calculated 
    # Multiply by -D 
    cyad.smul(P, +D, P, t) 
    
    # Now calculate the exponential and finish 
    cyad.smul(dum, -a3, Q1, t) 
    # use Q3 as a dummy 
    cyad.exp(Q3, dum, F, temp + 11, t)
    
    cyad.mul(v, P, Q3, t) # done!
    
    
    return 