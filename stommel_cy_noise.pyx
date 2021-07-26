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
                    mode='c'] noiseT = np.random.normal(loc=0.0,scale=1.0, size=(N*loop))
    cdef np.ndarray[DT, ndim=1, negative_indices=False,
                    mode='c'] noiseS = np.random.normal(loc=0.0,scale=1.0, size=(N*loop))
    cdef double ul0 = I[0]
    cdef double ul1 = I[1]
    cdef double usum0 = 0.
    cdef double usum1 = 0.
    cdef double k0, k1
    cdef double loop_float = float(loop)
    for n in xrange(N):
        usum0 = 0.
        usum1 = 0.
        for i in xrange(loop):
            k0 = dt*params[3]*(params[0]+ramp[loop*n+i]-ul0-fabs(ul0-ul1)*ul0)
            k1 = dt*params[3]*(params[1]-params[2]*ul1-fabs(ul0-ul1)*ul1)
            ul0 = ul0 + k0 + params[4]*sqrtdt*noiseT[loop*n+i]
            ul1 = ul1 + k1 + params[5]*sqrtdt*noiseS[loop*n+i]
            usum0+=ul0
            usum1+=ul1
        u0[n] = usum0/loop_float
        u1[n] = usum1/loop_float
    return u0, u1

cpdef heaviside(x):
    return 0.5*(copysign(1,x) + 1)
