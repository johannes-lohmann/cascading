import numpy as np
cimport numpy as np
from libc.math cimport tanh
from libc.math cimport copysign
from libc.math cimport fabs
cimport cython
ctypedef np.float64_t DT
cpdef solver(np.ndarray[DT, ndim=1, negative_indices=False, mode='c'] I, int N, int loop, double dt, np.ndarray[DT, ndim=1, negative_indices=False, 
                      mode='c'] params, np.ndarray[DT, ndim=1, negative_indices=False, mode='c'] ramp):
    cdef double sqrtdt = np.sqrt(dt)
    cdef int n, i
    cdef np.ndarray[DT, ndim=1, negative_indices=False, 
                    mode='c'] u0 = np.zeros(N)
    cdef np.ndarray[DT, ndim=1, negative_indices=False, 
                    mode='c'] u1 = np.zeros(N)
    cdef np.ndarray[DT, ndim=1, negative_indices=False, 
                    mode='c'] u2 = np.zeros(N)
    cdef np.ndarray[DT, ndim=1, negative_indices=False,
                    mode='c'] noise0 = np.random.normal(loc=0.0,scale=1.0, size=(N*loop))
    cdef np.ndarray[DT, ndim=1, negative_indices=False,
                    mode='c'] noise1 = np.random.normal(loc=0.0,scale=1.0, size=(N*loop))
    cdef np.ndarray[DT, ndim=1, negative_indices=False,
                    mode='c'] noise2 = np.random.normal(loc=0.0,scale=1.0, size=(N*loop))
    cdef double ul0 = I[0]
    cdef double ul1 = I[1]
    cdef double ul2 = I[2]
    cdef double k0, k1, k2
    for n in xrange(N):
        for i in xrange(loop):
            k0 = dt*(-1+params[0]*tanh(ul0/params[1])+(params[3]*heaviside(ul0)-params[4])*ul0 +params[6]-params[2]+params[5]+ramp[loop*n+i])
            k1 = dt*params[11]*(params[7]-params[10]*heaviside(ul0)*ul0-ul1-fabs(ul1-ul2)*ul1)
            k2 = dt*params[11]*(params[8]-params[9]*ul2-fabs(ul1-ul2)*ul2)
            ul0 = ul0 + k0 + params[12]*sqrtdt*noise0[loop*n+i]
            ul1 = ul1 + k1 + params[11]*params[13]*sqrtdt*noise1[loop*n+i]
            ul2 = ul2 + k2 + params[11]*params[14]*sqrtdt*noise2[loop*n+i]
        u0[n] = ul0
        u1[n] = ul1
        u2[n] = ul2
    return u0, u1, u2

cpdef heaviside(x):
    return 0.5*(copysign(1,x) + 1)

