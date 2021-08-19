#cython: boundscheck=False, wraparound=False, nonecheck=False
#
#
#
# Cython extension code for ndbasis 
# 

def _structured_op_double(double [:,:,:] X, double [:,:,:,:] Y, double [:,:] W, 
                          int [:] idx_op, int [:] idx_sub):
    
    cdef:
        size_t npre = X.shape[0]
        size_t Nb = X.shape[1]
        size_t npost = X.shape[2]
        
        size_t nq = W.shape[0]
        size_t nb = W.shape[1]
        
        size_t nsub = Y.shape[2] 
        
        size_t ipre,ipost,i,m
        int iop, isub 
        
        double X_val
        
    #
    #
    # X is a (npre, Nb, npost) sized input array
    #
    # Y is a (npre, nq, nsub, npost) sized output array (pre-allocated)
    #
    # initialize Y to zero
    for ipre in range(npre):
        for m in range(nq):
            for isub in range(nsub):
                for ipost in range(npost):
                    Y[ipre,m,isub,ipost] = 0.0 
    #
    #
    # Apply operator to structured basis
    #
    
    for i in range(Nb):
        iop = idx_op[i]
        isub = idx_sub[i]
        
        for ipre in range(npre):
            for ipost in range(npost):
                # For each basis function i
                # idx_op[i] is the index of the current basis factor and
                # idx_sub[i] is the index in the sub-structure
                X_val = X[ipre,i,ipost]
                
                for m in range(nq):
                    Y[ipre, m, isub, ipost] += W[m,iop] * X_val
    
    return 