#cython: boundscheck=False, wraparound=False, nonecheck=False, language_level=3
# 
#


##########################
# Define PES parameters
include "maldefs.pxi"
##########################

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
        size_t i,j,k
        size_t nd = dX.shape[0]
        size_t n = dX.shape[2] # the number of geometries
        int N = 9           # the number of atoms
        int ndist = 36      # the number of pairs 
        
        
        double **dx = cyad.malloc2d(3*N, nd)  # Cartesian geometries 
        double **dr = cyad.malloc2d(ndist, nd) # Pair distances 
        
        size_t tablesize = table.shape[1] 
        size_t *idxZ = <size_t *>malloc(tablesize * sizeof(size_t))
        size_t *idxX = <size_t *>malloc(tablesize * sizeof(size_t))
        size_t *idxY = <size_t *>malloc(tablesize * sizeof(size_t))
        
        cdef cyad.adtab tab = cyad.adtab(order, nd, tablesize, idxZ, idxX, idxY)
        
        size_t tempsize = 9
        double **temp = cyad.malloc2d(tempsize, nd) # workspace 
        double *Ftemp = cyad.malloc1d(order + 1)    # workspace 
        
        # Work space for gaussian distance inputs 
        
        (double *)[3] r 				
        double[3] x0
        double[3] alpha
        double D,s
        int ndim
        
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
        
        # Calculate pair distances 
        calc_dist(dx, dr, <size_t> N, Ftemp, temp, &tab)
        
        
        # Calculate potential
        #
        # Initialize derivative array 
        for k in range(nd):
            dV[k,0,i] = 0.0 
            
        # Add shift value 
        dV[0,0,i] = shift 
        
        #
        # Calculate zeroth-order Morse potentials
        #
        for k in range(nd):
            temp[0][k] = 0.0 
            
        for j in range(nmorse):
            
            calc_morse(temp[0], morse[j][0], morse[j][1], morse[j][2], dr[imorse[j]-1],
                        Ftemp, temp+1, &tab)
            for k in range(nd):
                dV[k,0,i] += temp[0][k]
            
        
        #
		# Calculate distributed Gaussian correction surface 
		#
        ndim = 1
        for j in range(ng1d):
            
            for k in range(ndim):
                r[k] = dr[ig1d[j*ndim + k] - 1]
                x0[k] = g1d[j*4 + k]
                alpha[k] = g1d[j*4 + ndim+k]
                
            D = g1d[j*4 + 2*ndim]
            s = g1d[j*4 + 2*ndim+1]
                
            calc_gauss(temp[0], ndim, r, x0, alpha, D, s, Ftemp, temp+1, &tab)
            for k in range(nd):
                dV[k,0,i] += temp[0][k] 
            
        ndim = 2
        for j in range(ng2d):
            
            for k in range(ndim):
                r[k] = dr[ig2d[j*ndim + k] - 1]
                x0[k] = g2d[j*6 + k]
                alpha[k] = g2d[j*6 + ndim+k]
                
            D = g2d[j*6 + 2*ndim]
            s = g2d[j*6 + 2*ndim+1]
 			
            calc_gauss(temp[0], ndim, r, x0, alpha, D, s, Ftemp, temp+1, &tab)
            for k in range(nd):
                dV[k,0,i] += temp[0][k] 
                
        ndim = 3
        for j in range(ng3d):
            
            for k in range(ndim):
                r[k] = dr[ig3d[j*ndim + k] - 1]
                x0[k] = g3d[j*8 + k]
                alpha[k] = g3d[j*8 + ndim+k]
                
            D = g3d[j*8 + 2*ndim]
            s = g3d[j*8 + 2*ndim+1]
 			
            calc_gauss(temp[0], ndim, r, x0, alpha, D, s, Ftemp, temp+1, &tab)
            for k in range(nd):
                dV[k,0,i] += temp[0][k] 

    
        # Convert from hartree to cm-1 
        for k in range(nd):
            dV[k,0,i] *= 219474.63136319697 
            
        
    cyad.free2d(dx, 3*N)
    cyad.free2d(dr, ndist) 
    free(idxZ)
    free(idxY)
    free(idxX)
    cyad.free2d(temp, tempsize)
    cyad.free1d(Ftemp) 
    
    return 


cdef void calc_dist(double **dx, double **dr, size_t natoms, double *F, double **temp,  cyad.adtab *t):
    # Input
    #   x in Angstroms
    #
    # Output
    #   dist in bohr
    
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
            
            
            cyad.sqrt(dr[idx], temp[3], F, temp + 4,  t)
            
            # Convert from Angstrom to bohr 
            cyad.smul(dr[idx], 1.8897261246257702, dr[idx], t)
            
            idx += 1 
    return 

cdef void calc_morse(double *v, double re, double alpha, double De, double *dr,
                      double *F, double **temp, cyad.adtab *t):
    # Calculate Morse function 
    # 
    # V(r) = De**2 * (1 - exp(-alpha * (r - re)))**2
    # cdef double v 
    # 
    # F ... (k+1) work
    # temp ... (4,nd) work
    #
    cdef size_t i
    
    # temp ... nd 
    
    
    # temp <-- re - r 
    for i in range(t.nd):
        temp[0][i] = -dr[i] 
    temp[0][0] += re 
    
    # temp *= alpha 
    cyad.smul(temp[0], alpha, temp[0], t)
    
    # v <-- exp(temp) - 1.0 
    cyad.exp(v, temp[0], F, temp + 1, t)
    v[0] -= 1.0 
    
    # temp = De * v
    cyad.smul(temp[0], De, v, t) 
    
    # v = temp * temp 
    cyad.mul(v, temp[0], temp[0], t)
    
    return

cdef void calc_gauss(double *v, int ndim, double **r, double *x, double *alpha, double D, double shift,
                      double *F, double **temp, cyad.adtab *t):
 	# Calculate Gaussian function
 	#
 	# V(r) = D * ( exp(-0.5 * a * rx**2) - shift )
    #
    # F ... (k+1) workspace 
    # temp (6,nd)
    
    cdef size_t i,k 
 	
    
    for k in range(t.nd):
        temp[2][k] = 0.0 # temp2 = 0 
    
    for i in range(ndim):
        #
        # temp0 <-- r[i] - x[i] 
        for k in range(t.nd):
            temp[0][k] = r[i][k]
        temp[0][0] -= x[i] 
        
        # temp1 <-- dr*dr
        cyad.mul(temp[1], temp[0], temp[0], t) 
        
        # temp2 += -0.5 * alpha[i] * dr*dr
        cyad.smulacc(temp[2], -0.5 * alpha[i], temp[1], t)
    
    cyad.exp(temp[0], temp[2], F, temp+3, t) # temp0 = exp[-alpha * (...)]
    temp[0][0] -= shift # - shift 
    
    cyad.smul(v, D, temp[0], t) 
    
    return 
