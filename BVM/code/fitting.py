# functions for fitting parameters for Measurement instances 
import model_utils as utils
from simulate import *
import scipy.optimize
import scipy.interpolate
import time
import numpy as np

    
# A function that calculates the objective to minimize in the parameter estimation
def estimation_objective(params, sim, f0_init=1, version='RMS-growth', meas_err=0.05):
    a = list(params)
    lb, ub = utils.get_bounds(type(params))
    if np.any((a < lb) | (a > ub)):
        return 1e100

    valid_meas = sim.data != -1
    log_meas = np.zeros_like(sim.data, dtype=float)
    log_meas[valid_meas] = np.log(sim.data[valid_meas])
    # WARNING hacky way to find n0
    growth_calc = utils.calc_meas_mat(sim.type, params, f0_init, sim.data[:, 0]).data
    log_calc = np.log(growth_calc)
    
    sum_of_sq = 0
    if version == 'RMS-growth':
        derr = log_meas[valid_meas] - log_calc[valid_meas]
        sum_of_sq = np.sum(derr**2)
    elif version == 'RMS-growthrate': # TODO: retest option after -1 measurement change
        vpair = valid_meas[:, 1:] & valid_meas[:, :-1]
        derr = (np.diff(log_meas) - np.diff(log_calc))[vpair]
        sum_of_sq = np.sum(derr**2)
    else:
        raise Exception("version not in [RMS-growth, RMS-growthrate]") # TODO: change option names

    return sum_of_sq

# a function to fit the parameters of a certain parameter regeme
# to a simulation or experiment. It uses the l-bfgs-b minimizer
# and does several basin hops to make sure it is not getting stuck 
# in a local minima
# it also takes in a strategy to deal with the initial f0. "equilib" allways assumes
# that it is equal to the equilibrium f0 for the parameters it is testing,
# given takes a given f0 (for example, the true value or 1) and "fit" allows one 
# to fit the value
def fit_params_log_growth_pulsed(sim, param_type, n_hops, meas_error, 
                                 liklihood_vers, f0_strat, 
                                 f0_init = 1):
    if f0_strat not in ["equilib", "given", "fit"]:
        raise Exception("f0_strat not in [equilib, given, fit]")
    if f0_strat == "given":
        obj = lambda a : \
                estimation_objective(param_type(*a), sim, version=liklihood_vers, 
                                     meas_err=meas_error, f0_init=f0_init)
    elif f0_strat == "equilib":
        obj = lambda a : \
                estimation_objective(param_type(*a), sim, version=liklihood_vers, 
                                     meas_err=meas_error, f0_init=utils.equilibf0(param_type(*a)))
    else:
        obj = lambda a : \
                estimation_objective(param_type(*a[:-1]), sim, version=liklihood_vers, 
                                     meas_err=meas_error, f0_init=a[-1])

    if f0_strat == "fit":
        lb, ub = utils.get_bounds(param_type) # lower and upper bounds for variables
        lb = lb + [0.0001]
        ub = ub + [0.9999]
        bounds = scipy.optimize.Bounds(lb, ub)
    else:
        lb, ub = utils.get_bounds(param_type) # lower and upper bounds for variables
        bounds = scipy.optimize.Bounds(lb, ub)
    
    x0 = np.random.uniform(low=lb, high=ub) # initial guess

    minimizer_kwargs = {
        'method': 'L-BFGS-B',
        'bounds': bounds,
        'options': {
            'ftol': 1e-12,
            'gtol': 1e-8,
            'maxiter': 50000,
            'maxfun': 50000,
            'maxcor': 20,
            'iprint': 2
        }
    }

    result = scipy.optimize.basinhopping(
        func=obj,
        x0=x0,
        niter=n_hops,  
        minimizer_kwargs=minimizer_kwargs,
        disp=True
    )
    x = result.x

    if f0_strat == "fit":
        return (param_type(*x[:-1]), x[-1])
    elif f0_strat == "equilib": 
        return (param_type(*x), utils.equilibf0(param_type(*x)))
    else:
        return (param_type(*x), f0_init)


