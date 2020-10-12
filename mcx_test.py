import mcx
import numpy as np
import timeit

num_paths = 200000
num_steps = 150
dt = 1/150

def test():
    dW_S = np.random.normal(size=(num_paths,num_steps))
    dW_X = np.random.normal(size=(num_paths,num_steps))

    S0 = 100.00; r = 0; q = 0; nu0 = 0.16**2; kappa = 1.0; theta = 0.16**2; xi = 0.30; rho = -0.60

    paths = np.array(mcx.heston_step(S0, nu0, kappa, theta, xi, rho, dt, r, q, num_paths, num_steps, dW_S, dW_X))
    print(np.mean(np.maximum(90.0 - paths[:,-1],0)))

if __name__ == '__main__':
    times = timeit.repeat(stmt=test,repeat=5,number=1)
    print('%0.2f s ± %0.2f s per loop'%(np.mean(times),np.std(times)))
