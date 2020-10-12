"""
Author: Isaac Drachman
Date:   12/15/2019
"""

import numpy as np
import copy
import mcx

# Feller condition for Heston to ensure positive variance process
def feller(kappa,theta,xi):
    return 2*kappa*theta > xi**2

# Inverse transform sampling from https://en.wikipedia.org/wiki/Poisson_distribution#Generating_Poisson-distributed_random_variables
def inverse_transform_sampling(lam,u):
    x = 0
    p = np.exp(-lam)
    s = p
    while u > s:
        x += 1
        p *= lam/x
        s += p
    return x

class Generator:
    def __init__(self, params):
        self.params = params
    def generate(self, num_paths, num_steps, dt):
        return np.zeros((num_paths,num_steps))
    def pertubate(self, param, to, shift='ovr'):
        new_gen = copy.deepcopy(self)
        if shift == 'ovr':
            new_gen.params[param] = to
        elif shift == 'abs':
            new_gen.params[param] += to
        elif shift == 'rel':
            new_gen.params[param] *= (1+to)
        return new_gen

class GBM_gen(Generator):
    # Required params: S0,r,q,sigma
    def __init__(self, params):
        Generator.__init__(self, params)
    def gen_rands(self, num_paths, num_steps):
        return np.random.normal(size=(num_paths,num_steps))
    def generate(self, num_paths, num_steps, dt, rands=[]):
        S0,r,q,sigma = self.params['S0'],self.params['r'],self.params['q'],self.params['sigma']

        noise = np.random.normal(size=(num_paths,num_steps)) if len(rands) == 0 else rands

        """
        # Original native python
        paths = np.zeros((num_paths, num_steps)) + S0
        for t in range(1,num_steps):
            paths[:,t] = paths[:,t-1]*(1 + (r-q)*dt + sigma*np.sqrt(dt)*noise[:,t])
        """

        return np.array(mcx.gbm_step(S0, r, q, sigma, dt, num_paths, num_steps, noise))

class GBMJD_gen(Generator):
    # Required params: S0,r,q,sigma,jump,lambda
    def __init__(self, params):
        Generator.__init__(self, params)
    def gen_rands(self, num_paths, num_steps):
        #return np.random.normal(size=(num_paths,num_steps)), np.random.uniform(size=(num_paths,num_steps))
        return []
    def generate(self, num_paths, num_steps, dt, rands=[]):
        paths = Generator.generate(self, num_paths, num_steps, dt)

        S0,r,q,sigma = self.params['S0'],self.params['r'],self.params['q'],self.params['sigma']
        jump,lam = self.params['jump'],self.params['lambda']

        #rands = self.gen_rands(num_paths, num_steps) if len(rands) == 0 else rands
        #noise = sigma*np.sqrt(dt)*rands[0]
        #poiss = np.vectorize(inverse_transform_sampling)(lam*dt, rands[1])
        noise = np.random.normal(size=(num_paths, num_steps))
        poiss = np.random.poisson(lam*dt, size=(num_paths, num_steps)).astype(np.float64)

        """
        # Original native python
        paths = np.zeros((num_paths,num_steps)) + S0
        for t in range(1,num_steps):
            paths[:,t] = paths[:,t-1]*(1 + (r-q)*dt + sigma*np.sqrt(dt)*noise[:,t] + jump*poiss[:,t])
        return paths
        """

        return np.array(mcx.gbmjd_step(S0, r, q, sigma, jump, dt, num_paths, num_steps, noise, poiss))

class OUJD_gen(Generator):
    # Required params: S0,mu,theta,sigma,jump,lambda
    def __init__(self, params):
        Generator.__init__(self, params)
    def gen_rands(self, num_paths, num_steps):
        #return np.random.normal(size=(num_paths,num_steps)), np.random.uniform(size=(num_paths,num_steps))
        return []
    def generate(self, num_paths, num_steps, dt, rands=[]):
        S0,mu,theta,sigma = self.params['S0'],self.params['mu'],self.params['theta'],self.params['sigma']
        jump,lam = self.params['jump'],self.params['lambda']

        #rands = self.gen_rands(num_paths, num_steps) if len(rands) == 0 else rands
        #noise = sigma*np.sqrt(dt)*rands[0]
        #poiss = np.vectorize(inverse_transform_sampling)(lam*dt, rands[1])
        noise = np.random.normal(size=(num_paths,num_steps))
        poiss = np.random.poisson(lam*dt, size=(num_paths,num_steps)).astype(np.float64)

        """
        # Original native python
        paths = np.zeros((num_paths,num_steps)) + S0
        for t in range(1,num_steps):
            paths[:,t] = paths[:,t-1] + theta*(mu - paths[:,t-1])*dt + sigma*np.sqrt(dt)*noise[:,t] + jump*poiss[:,t]
        return paths
        """

        return np.array(mcx.oujd_step(S0, mu, theta, sigma, jump, dt, num_paths, num_steps, noise, poiss))

class Heston_gen(Generator):
    # Required params: S0,r,q,nu0,kappa,theta,xi,rho
    def __init__(self, params):
        Generator.__init__(self, params)
    # Generate appropriate shape/type random variables
    def gen_rands(self, num_paths, num_steps):
        return np.random.normal(size=(2,num_paths,num_steps))
    def generate(self, num_paths, num_steps, dt, rands=[]):
        S0,r,q,nu0 = self.params['S0'],self.params['r'],self.params['q'],self.params['nu0']
        kappa,theta,xi,rho = self.params['kappa'],self.params['theta'],self.params['xi'],self.params['rho']

        rands = self.gen_rands(num_paths, num_steps) if len(rands) == 0 else rands
        dW_S = rands[0]
        dW_X = rands[1]

        """
        # Original native python
        paths = np.zeros((num_paths,num_steps)) + S0
        dW_nu = rho*dW_S + np.sqrt(1 - rho**2)*dW_X
        nu = np.zeros((num_paths)) + nu0
        for t in range(1,num_steps):
            # Since the variance process is discrete, it can take on negative values
            # Use full truncation, i.e. v_t <- max(v_t,0)
            nu = nu + kappa*(theta - np.maximum(nu,0))*dt + xi*np.sqrt(np.maximum(nu,0)*dt)*dW_nu[:,t]
            paths[:,t] = paths[:,t-1]*(1 + (r-q)*dt + np.sqrt(np.maximum(nu,0)*dt)*dW_S[:,t])
        return paths
        """
        # Run cython routine
        return np.array(mcx.heston_step(S0,nu0,kappa,theta,xi,rho,dt,r,q,num_paths,num_steps,dW_S,dW_X))

class Option:
    def __init__(self, strike, expiry, PC, AE, val_params):
        self.strike = strike
        self.expiry = expiry
        self.PC = PC
        self.AE = AE
        self.val_params = val_params

    def payoff(self, S_T):
        return np.maximum(self.strike - S_T, 0) if self.PC == 'P' else np.maximum(S_T - self.strike, 0)

    def value(self, gen=None, paths=None):
        if gen is None and paths is None:
            raise Exception('either MCgen or paths required')

        # Instead of re-running the generator, check if we've been given paths already
        if paths is None:
            num_paths = self.val_params['num_paths']
            num_steps = self.val_params['num_steps']
            dt = self.expiry / num_steps

            paths = gen.generate(num_paths, num_steps, dt)
        values = self.payoff(paths[:,-1])
        # Discount the expected payoff
        return np.exp(-self.val_params['r']*self.expiry)*values.mean()

    def delta(self, gen, bump=0.01):
        value_S = self.value(gen)
        value_Splus = self.value(gen.pertubate('S0',bump,shift='rel'))
        value_Sminus = self.value(gen.pertubate('S0',-bump,shift='rel'))
        return ((value_Splus - value_S)/0.01 + (value_S - value_Sminus)/0.01)*0.5

    def time_series(self, gen, path):
        values = np.zeros(len(path))
        deltas = np.zeros(len(path))
        for t in range(len(path)):
            new_gen = gen.pertubate('S0',path[t],shift='ovr')
            values[t] = self.value(new_gen)
            deltas[t] = self.value(new_gen)
            self.val_params['num_steps'] -= 1
        return values, deltas
