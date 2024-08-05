#cython: boundscheck=False, wraparound=False, nonecheck=False, language_level=3
"""
Simple Cython implementation of forward AD

cyad_core.pxd 

Definition source file

"""

ctypedef struct adtab:
    int k              # Maximum derivative order
    size_t nd          # Number of derivatives
    size_t table_size  # Size of product table
    size_t *idxZ       # Result indices
    size_t *idxX       # Left factor index
    size_t *idxY       # Right factor index
    
    
cdef void mul(double *Z, double *X, double *Y, adtab *t)        # Multiply
cdef void mulacc(double *Z, double *X, double *Y, adtab *t)     # Multiply w/ accumulation
cdef void add(double *Z, double *X, double *Y, adtab *t)        # Addition
cdef void smul(double *Z, double s, double *X, adtab *t)        # Scalar multiplication
cdef void smulacc(double *Z, double s, double *X, adtab *t)     # Scalar multiplication w/ accumulation
cdef void sub(double *Z, double *X, double *Y, adtab *t)        # Subtraction 

cdef void sqrt(double *Z, double *X, double *F, double **temp, adtab *t)
cdef void exp(double *Z, double *X, double *F, double **temp, adtab *t)

cdef void chain1d(double *Z, double *X, double *F, double **temp, adtab *t)

cdef double *malloc1d(size_t n)
cdef double **malloc2d(size_t m, size_t n)
cdef void free1d(double *ptr)
cdef void free2d(double **ptr, size_t m)