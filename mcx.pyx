"""
Author: Isaac Drachman
Date:   12/15/2019

Cython implementation of the discrete steps for generating stochastic processes.
"""

cimport cython
import numpy as np
cimport numpy as np
from libc.math cimport sqrt as c_sqrt

cdef double c_max(double x):
    return x if x > 0.0 else 0.0

def gbm_step(float S0, float r, float q, float sigma, float dt, int num_paths, int num_steps,
             np.ndarray[np.float64_t, ndim=2] noise):
    cdef np.ndarray[np.float64_t, ndim=2] paths = np.zeros((num_paths, num_steps)) + S0
    cdef int t
    for t in range(1,num_steps):
        paths[:,t] = paths[:,t-1]*(1 + (r-q)*dt + sigma*np.sqrt(dt)*noise[:,t])
    return paths

@cython.boundscheck(False)
@cython.cdivision(True)
@cython.wraparound(False)
def heston_step(float S0, float nu0, float kappa, float theta, float xi, float rho, float dt, float r, float q, int num_paths, int num_steps,
                double[:,:] dW_S, double[:,:] dW_X):
    cdef double[:,:] paths = np.full((num_paths,num_steps), S0)
    cdef double[:] nu = np.full((num_paths), nu0)
    cdef double dW_nu
    cdef Py_ssize_t t, idx
    for t in range(1,num_steps):
        for idx in range(num_paths):
            dW_nu = rho*dW_S[idx,t] + c_sqrt(1 - rho**2)*dW_X[idx,t]
            nu[idx] = c_max(nu[idx] + kappa*(theta - nu[idx])*dt + xi*c_sqrt(nu[idx]*dt)*dW_nu)
            paths[idx,t] = paths[idx,t-1]*(1 + (r-q)*dt + c_sqrt(nu[idx]*dt)*dW_S[idx,t])
    return paths

@cython.boundscheck(False)
@cython.cdivision(True)
@cython.wraparound(False)
def oujd_step(float S0, float mu, float theta, float sigma, float jump, float dt, int num_paths, int num_steps,
              double[:,:] noise, double[:,:] poiss):
    cdef double[:,:] paths = np.full((num_paths,num_steps),S0)
    cdef Py_ssize_t t, idx
    for t in range(1,num_steps):
        for idx in range(num_paths):
            paths[idx,t] = paths[idx,t-1] + theta*(mu - paths[idx,t-1])*dt + sigma*c_sqrt(dt)*noise[idx,t] + jump*poiss[idx,t]
    return paths

def gbmjd_step(float S0, float r, float q, float sigma, float jump, float dt, int num_paths, int num_steps,
               double[:,:] noise, double[:,:] poiss):
    cdef double[:,:] paths = np.full((num_paths,num_steps),S0)
    cdef Py_ssize_t t, idx
    for t in range(1,num_steps):
        for idx in range(num_paths):
            paths[idx,t] = paths[idx,t-1]*(1 + (r-q)*dt + sigma*c_sqrt(dt)*noise[idx,t] + jump*poiss[idx,t])
    return paths
