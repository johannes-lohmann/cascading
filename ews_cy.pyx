import numpy as np
cimport numpy as np
from libc.math cimport sqrt
cimport cython
ctypedef np.float64_t DT

cpdef estimated_ac1(np.ndarray[DT, ndim=1, negative_indices=False, mode='c'] x, int lag):
    cdef DT results
    result = 1/cy_mult_sum(x,x)*cy_ac(x,lag)
    return result

cpdef estimated_ac1_truemean(np.ndarray[DT, ndim=1, negative_indices=False, mode='c'] x, double bmean):
    cdef int lag = 50
    cdef DT results
    result = 1/cy_mult_sum(x,x)*cy_ac_truemean(x,lag,bmean)
    return result

cpdef estimated_cc(np.ndarray[DT, ndim=1, negative_indices=False, mode='c'] x, np.ndarray[DT, ndim=1, negative_indices=False, mode='c'] y):

    cdef DT results
    result = 1/(sqrt(cy_var(x))*sqrt(cy_var(x)))*cy_mult_sum(x,y)
    return result

cpdef cy_mult_sum(np.ndarray[DT, ndim=1, negative_indices=False,
                    mode='c'] b, np.ndarray[DT, ndim=1, negative_indices=False, mode='c'] c):
    cdef int N = b.shape[0]
    cdef DT a = 0
    cdef int i

    cdef DT bmean = b[0]
    for i in xrange(1,N):
        bmean += b[i]
    bmean = bmean/N

    cdef DT cmean = c[0]
    for i in xrange(1,N):
        cmean += c[i]
    cmean = cmean/N 

    for i in xrange(0,N):
        a += (b[i]-bmean)*(c[i]-cmean)
    return a/N

cpdef cy_ac(np.ndarray[DT, ndim=1, negative_indices=False,
                    mode='c'] b, int lag):
    cdef int N = b.shape[0]
    cdef DT a = 0
    cdef int i

    cdef DT bmean = b[0]
    for i in xrange(1,N):
        bmean += b[i]
    bmean = bmean/N

    for i in xrange(0,N-lag):
        a += (b[i]-bmean)*(b[i+lag]-bmean)
    return a/(N-lag)



cpdef cy_asym(np.ndarray[DT, ndim=1, negative_indices=False,
                    mode='c'] b, int lag):
    cdef int N = b.shape[0]
    cdef DT a = 0
    cdef int i

    cdef DT bmean = b[0]
    for i in xrange(1,N):
        bmean += b[i]
    bmean = bmean/N

    for i in xrange(0,N-lag):
        a += ((b[i+lag]-bmean) - (b[i]-bmean))**3
    return a/(N-lag)

cpdef cy_higher_ac(np.ndarray[DT, ndim=1, negative_indices=False,
                    mode='c'] b):
    cdef int N = b.shape[0]
    cdef DT a = 0
    cdef int i

    cdef DT bmean = b[0]
    for i in xrange(1,N):
        bmean += b[i]
    bmean = bmean/N

    for i in xrange(0,N-2):
        a += (b[i+2]-bmean)*(b[i+1]-bmean)*(b[i]-bmean)
    return a/(N-2)



cpdef cy_ac_truemean(np.ndarray[DT, ndim=1, negative_indices=False,
                    mode='c'] b, int lag, double bmean):
    cdef int N = b.shape[0]
    cdef DT a = b[0]*b[lag]
    cdef int i

    for i in xrange(1,N-lag):
        a += (b[i]-bmean)*(b[i+lag]-bmean)
    return a/(N-lag)

cpdef cy_var(np.ndarray[DT, ndim=1, negative_indices=False,
                    mode='c'] b):
    cdef int N = b.shape[0]
    cdef DT a = 0
    cdef int i
    cdef DT bmean = b[0]
    for i in xrange(1,N):
        bmean += b[i]
    bmean = bmean/N

    for i in xrange(0,N):
        a += (b[i]-bmean)*(b[i]-bmean)
    return a/(N-1)

cpdef cy_fourth(np.ndarray[DT, ndim=1, negative_indices=False,
                    mode='c'] b):
    cdef int N = b.shape[0]
    cdef DT a = 0
    cdef int i
    cdef DT bmean = b[0]
    for i in xrange(1,N):
        bmean += b[i]
    bmean = bmean/N

    for i in xrange(0,N):
        a += (b[i]-bmean)*(b[i]-bmean)*(b[i]-bmean)*(b[i]-bmean)
    return a/(N-1)

cpdef cy_skew(np.ndarray[DT, ndim=1, negative_indices=False,
                    mode='c'] b):
    cdef int N = b.shape[0]
    cdef DT a = b[0]*b[0]
    cdef DT c = b[0]*b[0]*b[0]
    cdef int i
    cdef DT bmean = b[0]
    for i in xrange(1,N):
        bmean += b[i]
    bmean = bmean/N

    for i in xrange(1,N):
        a += (b[i]-bmean)*(b[i]-bmean)
        c += (b[i]-bmean)*(b[i]-bmean)*(b[i]-bmean)
    return c/(a**(3./2))*(N**(3./2)/N)

cpdef cy_var_truemean(np.ndarray[DT, ndim=1, negative_indices=False,
                    mode='c'] b, double bmean):
    cdef int N = b.shape[0]
    cdef DT a = b[0]*b[0]
    cdef int i

    for i in xrange(1,N):
        a += (b[i]-bmean)*(b[i]-bmean)
    return a/(N-1)


