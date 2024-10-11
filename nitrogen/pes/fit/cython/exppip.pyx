#cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True, language_level=3
# 
"""
exppip.pyx

Cython implementation of simple exponential PIP surface
with forward AD.



"""
cimport nitrogen.autodiff.cyad.cyad_core as cyad

from libc.stdlib cimport malloc, free 

def evalexppip(double [:,:,:] dX, double [:,:,:] dV, int order, int [:,:] table,
               double a, int natoms, int maxmon, double [:] coeff, int [:,:] terms):

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
        
    double a 
        The Morse parameter
    
    int natoms
        The number of atoms 
    
    int maxmon
        The maximum monomial power 
    
    double [nt] coeff 
        The expansion coefficients 
        
    int [nt,ny] terms
        The expansion term powers. `ny` = `natoms`(`natoms`-1)/2 is the number of
        atomic pairs. 
        
    """
    
    cdef:
        size_t i,j,k,m 
        size_t nd = dX.shape[0] # the number of derivatives
        size_t n = dX.shape[2]  # the number of evaluation points
        
        int N = natoms          # the number of atoms 
        int ny = (natoms*(natoms-1)) // 2 # the number of atomic pairs 

        
        double **dx = cyad.malloc2d(3*N, nd)   # Cartesian geometries 
        double **dy = cyad.malloc2d(ny, nd)    # Pair distance functions
        double **dyn = cyad.malloc2d(ny*(maxmon+1), nd) # Powers of y 
        
        size_t tablesize = table.shape[1] 
        size_t *idxZ = <size_t *>malloc(tablesize * sizeof(size_t))
        size_t *idxX = <size_t *>malloc(tablesize * sizeof(size_t))
        size_t *idxY = <size_t *>malloc(tablesize * sizeof(size_t))
        
        cdef cyad.adtab tab = cyad.adtab(order, nd, tablesize, idxZ, idxX, idxY)
        
        size_t tempsize = 9
        double **temp = cyad.malloc2d(tempsize, nd) # workspace 
        double *Ftemp = cyad.malloc1d(order + 1)    # workspace 
        
        double *tA = cyad.malloc1d(nd)
        double *tB = cyad.malloc1d(nd)
        double *prod1 
        double *prod2 
        double *tempprod 
        
        int p 
        
        size_t nt = terms.shape[0] 
        # terms.shape[1] should equal `ny`

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
        
        #
        # Calculate y variables
        #
        calc_y(dx, dy, N, a, Ftemp, temp, &tab) 
        
        # Calculate their powers
        calc_yn(dy, dyn, ny, maxmon, &tab)
        
        # Calculate expansion 
        #
        # Initialize derivative array 
        for k in range(nd):
            dV[k,0,i] = 0.0 
            
        # Loop through terms 
        for j in range(nt): 
            
            # Initialize `prod` to unity
            prod1 = tA 
            prod2 = tB 
            
            prod1[0] = 1.0 
            for k in range(1,nd):
                prod1[k] = 0.0
                
            # Compute product 
            for m in range(ny):
                #
                # If the power of y[m] is greater than 0,
                # compute the product.
                #
                p = terms[j,m]
                
                if p > 0 :
                    # prod2 <-- prod1 * y[m]**p 
                    cyad.mul(prod2, prod1, dyn[m*(maxmon+1) + p], &tab)
                    # swap prod2 and prod1 pointers 
                    tempprod = prod1 
                    prod1 = prod2
                    prod2 = tempprod 
                    #
                    # prod1 now points to the current product
                    # prod2 points to the other temp space 
                    
            
            # Scale by coefficient 
            #
            # prod <-- coeff[j] * prod 
            cyad.smul(prod1, coeff[j], prod1, &tab) 
            
            # Add term to sum 
            for k in range(nd):
                dV[k,0,i] += prod1[k] 
            
        
    cyad.free2d(dx, 3*N)
    cyad.free2d(dy, ny) 
    cyad.free2d(dyn, ny*(maxmon+1))
    free(idxZ)
    free(idxY)
    free(idxX)
    cyad.free1d(tA) 
    cyad.free1d(tB) 
    cyad.free2d(temp, tempsize)
    cyad.free1d(Ftemp) 
    
    return 

def evalexppipterms(double [:,:,:] dX, double [:,:,:] dV, int order, int [:,:] table,
                    double a, int natoms, int maxmon, int [:,:] terms):

    """
    Evaluate individual terms of the exp-pip expansion.

    Parameters
    ----------
    double [nd,3*N,n] dX
        The 3*N Cartesian coordinates at `n` geometries
        
    double [nd,nt,n] dV
        The energy output for each term.
        
    int order 
        Derivative order 
        
    int [3,tablesize] table
        Product table 
        
    double a 
        The Morse parameter
    
    int natoms
        The number of atoms 
    
    int maxmon
        The maximum monomial power 
        
    int [nt,ny] terms
        The expansion term powers. `ny` = `natoms`(`natoms`-1)/2 is the number of
        atomic pairs. 
        
    """
    
    cdef:
        size_t i,j,k,m
        size_t nd = dX.shape[0] # the number of derivatives
        size_t n = dX.shape[2]  # the number of evaluation points
        
        int N = natoms          # the number of atoms 
        int ny = (natoms*(natoms-1)) // 2 # the number of atomic pairs 

        
        double **dx = cyad.malloc2d(3*N, nd)   # Cartesian geometries 
        double **dy = cyad.malloc2d(ny, nd)    # Pair distance functions
        double **dyn = cyad.malloc2d(ny*(maxmon+1), nd) # Powers of y 
        
        size_t tablesize = table.shape[1] 
        size_t *idxZ = <size_t *>malloc(tablesize * sizeof(size_t))
        size_t *idxX = <size_t *>malloc(tablesize * sizeof(size_t))
        size_t *idxY = <size_t *>malloc(tablesize * sizeof(size_t))
        
        cdef cyad.adtab tab = cyad.adtab(order, nd, tablesize, idxZ, idxX, idxY)
        
        size_t tempsize = 9
        double **temp = cyad.malloc2d(tempsize, nd) # workspace 
        double *Ftemp = cyad.malloc1d(order + 1)    # workspace 
        
        double *tA = cyad.malloc1d(nd)
        double *tB = cyad.malloc1d(nd)
        double *prod1 
        double *prod2 
        double *tempprod 
        
        int p 
        
        size_t nt = terms.shape[0] 
        # terms.shape[1] should equal `ny`

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
        
        #
        # Calculate y variables
        #
        calc_y(dx, dy, N, a, Ftemp, temp, &tab) 
        
        # Calculate their powers
        calc_yn(dy, dyn, ny, maxmon, &tab)
        
        # Calculate expansion 
        #
        #
        # Initialize derivative array 
        # for k in range(nd):
        #     for j in range(nt):
        #         dV[k,j,i] = 0.0 
        # (No initialization needed b/c terms
        #  will be assigned.)
        #
            
        # Loop through terms 
        for j in range(nt): 
            
            # Initialize `prod` to unity
            prod1 = tA 
            prod2 = tB 
            
            prod1[0] = 1.0 
            for k in range(1,nd):
                prod1[k] = 0.0
                
            # Compute product 
            for m in range(ny):
                #
                # If the power of y[m] is greater than 0,
                # compute the product.
                #
                p = terms[j,m]
                
                if p > 0 :
                    # prod2 <-- prod1 * y[m]**p 
                    cyad.mul(prod2, prod1, dyn[m*(maxmon+1) + p], &tab)
                    # swap prod2 and prod1 pointers 
                    tempprod = prod1 
                    prod1 = prod2
                    prod2 = tempprod 
                    #
                    # prod1 now points to the current product
                    # prod2 points to the other temp space 
                    
            
            # prod1 contains the term
            
            # Assign term to output
            for k in range(nd):
                dV[k,j,i] = prod1[k] 
            
        
    cyad.free2d(dx, 3*N)
    cyad.free2d(dy, ny) 
    cyad.free2d(dyn, ny*(maxmon+1))
    free(idxZ)
    free(idxY)
    free(idxX)
    cyad.free1d(tA) 
    cyad.free1d(tB) 
    cyad.free2d(temp, tempsize)
    cyad.free1d(Ftemp) 
    
    return


cdef void calc_y(double **dx, double **dy, int natoms, double a, 
                 double *F, double **temp, cyad.adtab *t):
    #
    # Input 
    #    x in Angstroms
    #
    # Output 
    #
    #    y = exp(-r/a)
    # 
    # The y's are ordered y_12, y_13, ... y_23, ... y_n-1,n
    
    # Workspace
    # F (k+1) double 
    # temp (7,nd) double 
    #
    cdef int i,j,idx
    cdef double mia = -1.0 / a
    
    idx = 0 
    for i in range(natoms):
        for j in range(i+1, natoms):
            
            cyad.sub(temp[0], dx[3*i+0], dx[3*j+0], t) # delta x
            cyad.sub(temp[1], dx[3*i+1], dx[3*j+1], t) # delta y 
            cyad.sub(temp[2], dx[3*i+2], dx[3*j+2], t) # delta z
            
            cyad.mul(temp[3], temp[0], temp[0], t) # dx * dx 
            cyad.mulacc(temp[3], temp[1], temp[1], t) # dy * dy 
            cyad.mulacc(temp[3], temp[2], temp[2], t) # dz * dz 
            # r2 = dx*dx + dy*dy + dz*dz 
            # 
            # temp[3] holds r**2 
            # temp[0] <-- sqrt[r**2]
            cyad.sqrt(temp[0], temp[3], F, temp + 4,  t)
            
            # temp[1] <--  -r/a
            cyad.smul(temp[1], mia, temp[0], t)
            
            # dy[idx] <-- exp(-r/a)
            cyad.exp(dy[idx], temp[1], F, temp + 4, t)
            
            idx += 1 
    
    return 

cdef void calc_yn(double **dy, double **dyn, int ny, int maxmon, cyad.adtab *t):
    #
    #
    # Calculate powers of y
    #
    # Input 
    #   dy ... (ny, nd) 
    # Output
    #   dyn ... (ny*(maxmon+1), nd)
    #
    #
    #  y[i]**m is dyn[i*(maxmon+1) + m]
    #
    cdef size_t i,m,k 
    cdef size_t nd = t.nd # The number of derivatives 
    
    # Initialize to zero 
    for i in range(ny * (maxmon+1)):
        for k in range(nd):
            dyn[i][k] = 0.0 
    
    # Calculate powers of y 
    for i in range(ny):
        #
        # For each y[i] 
        #
        #
        # The 0**th power is just unity 
        #
        dyn[i*(maxmon+1) + 0][0] = 1.0 
        
        # The 1**st power is just y[i] 
        if maxmon >= 1: 
            for k in range(nd):
                dyn[i*(maxmon+1) + 1][k] = dy[i][k]  
                
        # Higher powers are calculated as y[i]**(m-1) * y[i] 
        for m in range(1, maxmon):
            #
            # y[i]**(m+1) = y[i]**(m) * y[i] 
            #
            cyad.mul(dyn[i*(maxmon+1) + m + 1],
                     dyn[i*(maxmon+1) + m],
                     dyn[i*(maxmon+1) + 1],
                     t)
        
    
    return 
