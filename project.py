"""
Author: Isaac Drachman
Date:   12/15/2019
"""

import pandas as pd
import numpy as np
import scipy.optimize as opt
import mc
import sys

# Read csv data
def read_data(filename):
    options = pd.read_csv(filename)

    # Accounting rate in %, and compute q (i.e. div+borrow)
    # Compute mid-price for each option
    options.Rate /= 100
    options['Q'] = options.Rate - np.log(options.Fwd/options.Spot)/options.TTE
    options['Mid'] = 0.5*(options['Bid'] + options['Ask'])

    # Remove ITM options and those with no-bid
    options = options.loc[(options.Bid > 0.0) & (((options.Strike > options.Spot) & (options.CallPut == 'C')) |
                          ((options.Strike < options.Spot) & (options.CallPut == 'P')))]
    return options

# Build mc.Option objects out of each option
def data_to_options(data):
    return [mc.Option(row.Strike, row.TTE, row.CallPut, 'E', {'r':row.Rate}) for idx,row in data.iterrows()]

# Slice for given date/strikes
def select_options(options, date, strikes):
    return options.loc[(options.Date == date) & (options.Strike.isin(strikes))]

class Calibrator:
    def __init__(self, options, gen_class, val_params={}):
        self.options = options
        self.gen_class = gen_class
        self.val_params = val_params

        # Default MC specs
        if self.val_params == {}:
            self.val_params['num_paths'] = 100000
            self.val_params['num_steps'] = 150

        # Generate all the random variables upfront to speed up computation
        # For each iteration of new model params, same initial random variables are used
        self.rands = self.gen_class({}).gen_rands(self.val_params['num_paths'], self.val_params['num_steps'])

    # updates:     list/array of new params
    # update_keys: list of keys for each param being updated in 'updates'
    # model:       dictionary of model parameters (static and those being updated)
    # fixrands:    if True, keeps the random numbers generated when calibrator object was initialized
    def evaluate(self, updates, update_keys, model, fixrands=True):
        num_paths = self.val_params['num_paths']
        num_steps = self.val_params['num_steps']

        # Assumes same expiry for all options
        dt = self.options[0].expiry / num_steps

        # Update model params we are calibrating
        for idx,key in enumerate(update_keys):
            model[key] = updates[idx]

        # Value all options on the same set of paths for performance
        gen = self.gen_class(model)
        if fixrands:
            paths = gen.generate(num_paths,num_steps,dt,rands=self.rands)
        else:
            paths = gen.generate(num_paths,num_steps,dt)
        values = np.array([mc_opt.value(paths=paths) for mc_opt in self.options])
        return values

    # Compute one of 3 loss functions
    def loss(self, updates, update_keys, model, market, fixrands=True):
        # Adds the Feller constraint for DE since constraints not built-in
        if self.gen_class == mc.Heston_gen and self.val_params['optimizer'] == 'DE':
            if not mc.feller(updates[1],updates[2],updates[3]):
                return 10000

        # Price all the options
        px = self.evaluate(updates, update_keys, model, fixrands=fixrands)
        if self.val_params['loss'] == 'abserror':
            # Use the average absolute percentage error
            return np.mean(np.abs(100*(px/market - 1)))
        elif self.val_params['loss'] == 'pctinside':
            # Use the percentage within bid/ask
            return 1 - len(px[(px > market[0]) & (px < market[1])])/len(px)
        elif self.val_params['loss'] == 'mse':
            # Use the mean squared $ error
            return np.mean((px - market)**2)

    # Gradient descent
    def run_GD(self,x0,bounds,cons,model,update_keys,market,lossfunc='abserror'):
        self.val_params['optimizer'] = 'GD'
        self.val_params['loss'] = lossfunc
        res = opt.minimize(self.loss, x0, args=(update_keys, model, market), bounds=bounds, constraints=cons)
        return res['x']

    # Differential evolution
    def run_DE(self,bounds,model,update_keys,market,lossfunc='abserror',maxiter=50,popsize=15):
        self.val_params['optimizer'] = 'DE'
        self.val_params['loss'] = lossfunc
        # The workers=-1 flag uses multiprocessing, and required updating='deferred'
        # Remove the workers flag and change updating='immediate' if not using multiprocessing
        res = opt.differential_evolution(self.loss, bounds, args=(update_keys, model, market), maxiter=maxiter, popsize=popsize, disp=True, updating='deferred', workers=-1)
        return res['x']

    # Helper functions for generating bounds, conditions, and initial states for the various models
    # I tried to make the bounds somewhat general/wide, not specific to knowing the underlying is NDX
    # Order: ['nu0','kappa','theta','xi','rho']
    def heston_bounds(self):
        # initial vol: 1 to 50 vols, reversion: 0 to 5, mean: 1 to 50 vols, vol-of-vol: 10 to 150 vols, spot/vol correl: -1 to 0
        return [(0.01**2,0.50**2),(0,5),(0.01**2,0.50**2),(0.10,1.50),(-1,0)]
    def heston_initial(self):
        return [0.14**2, 0.9, 0.23**2, 0.30, -0.4]
    # Define the Feller constraint to ensure positive variance
    def heston_cons(self):
        return [{'type':'ineq', 'fun': lambda x: 2*x[1]*x[2] - x[3]**2}]

    # Order: ['sigma','jump','lambda']
    def gbmjd_bounds(self):
        # vol: 1 to 50 vols, jump size: -10 to +10%, jump rate: 0 to 10 /year
        return [(0.01,0.50),(-0.10,0.10),(0,10)]

    # Order: ['mu','sigma','theta','jump','lambda']
    def oujd_bounds(self):
        # mean: $8000 to 9500, vol: $100 to 300, reversion: 0 to 10, jump size: -$800 to +$800, jump rate: 0 to 10 /year
        return [(8000,9500),(100,3000),(0,10),(-800,800),(0,10)]

def setup_example(date):
    data = read_data('option_px.csv')

    # I've hardcoded the strikes to be fit
    # 25 options (wide range)
    #selection = select_options(data, date, [7500,7600,7800,7900,8000,8100,8200,8250,8275,8300,8325,8350,8375,8400,8425,8450,8475,8500,8525,8550,8600,8700,8800,8900,9000])
    # 18 options (tight range)
    #selection = select_options(data, date, [8000,8100,8200,8250,8275,8300,8325,8350,8375,8400,8425,8450,8475,8500,8525,8550,8600,8700])
    selection = select_options(data, date, [8125,8150,8175,8200,8225,8250,8275,8300,8325,8350,8375,8400,8425,8450,8475,8500,8525,8550,8575,8600,8625,8650,8675])
    options = data_to_options(selection)

    # Pull spot, risk-free, cost-of-carry, these are independent of model
    model = {'S0': selection.Spot.iloc[0], 'r': selection.Rate.iloc[0], 'q': selection.Q.iloc[0]}
    return selection, options, model

# Usage: python project.py 2019-12-06 heston [mse|abserror] [gd|de]
if __name__ == '__main__':
    selection, options, model = setup_example(sys.argv[1])
    lossfunc = 'abserror' if len(sys.argv) < 4 else sys.argv[3]
    algo = 'gd' if len(sys.argv) == 5 and sys.argv[4] == 'gd' else 'de'
    if sys.argv[2] == 'heston':
        calibrator = Calibrator(options, mc.Heston_gen)
        update_keys = ['nu0','kappa','theta','xi','rho']
        if algo == 'de':
            res = calibrator.run_DE(calibrator.heston_bounds(), model, update_keys, selection.Mid.values, lossfunc=lossfunc)
        elif algo == 'gd':
            res = calibrator.run_GD(calibrator.heston_initial(), calibrator.heston_bounds(), calibrator.heston_cons(), model, update_keys, selection.Mid.values, lossfunc=lossfunc)
    elif sys.argv[2] == 'gbmjd':
        calibrator = Calibrator(options, mc.GBMJD_gen)
        update_keys = ['sigma','jump','lambda']
        res = calibrator.run_DE(calibrator.gbmjd_bounds(), model, update_keys, selection.Mid.values, lossfunc=lossfunc)
    elif sys.argv[2] == 'oujd':
        calibrator = Calibrator(options, mc.OUJD_gen)
        update_keys = ['mu','sigma','theta','jump','lambda']
        res = calibrator.run_DE(calibrator.oujd_bounds(), model, update_keys, selection.Mid.values, lossfunc=lossfunc)

    # Build output, print some stats, and save to csv
    px  = calibrator.evaluate(res, update_keys, model)
    selection['predicted'] = px
    selection['error/px'] = np.abs(px - selection.Mid.values)/selection.Mid.values
    selection['error/spot'] = np.abs(px - selection.Mid.values)/selection.Spot.iloc[0]

    print('result: '+str(res))
    print('avg. %% error: %0.2f%%'%(100*selection['error/px'].mean()))
    print('avg. $ error: $%0.2f (%0.2f%% vs spot)'%((selection['error/px']*selection.Mid.values).mean(),100*selection['error/spot'].mean()))
    selection.to_csv('prices_%s_%s_%s_%s.csv'%(sys.argv[1],sys.argv[2],lossfunc,algo))
    pd.DataFrame([res],columns=update_keys).to_csv('results_%s_%s_%s_%s.csv'%(sys.argv[1],sys.argv[2],lossfunc,algo))
