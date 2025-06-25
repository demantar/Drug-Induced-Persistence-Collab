# functions for fitting parameters for Measurement instances 
import model_utils as utils
from simulate import *
import scipy.optimize
import scipy.interpolate
import time
import numpy as np

    
# A function that calculates the objective to minimize in the parameter estimation
def estimation_objective(params, sim, f0_init=1, version='RMS-growth', meas_err=0.05):
    begin = time.time()
    a = list(params)
    if np.any(np.array(a) > 1):
        return 1e100
    lb, ub = utils.get_bounds(type(params))
    if np.any((a < lb) | (a > ub)):
        return 1e100

    log_growth_meas = np.log(sim.data)
    # WARNING hacky way to find n0
    log_growth_calc = np.log(utils.calc_meas_mat(sim.type, params, f0_init, sim.data[:, 0]).data)
    
    sum_of_sq = 0
    if version == 'RMS-growth':
        sum_of_sq = np.nansum((log_growth_meas - log_growth_calc)**2, axis = None)
    elif version == 'RMS-growthrate':
        sum_of_sq = np.nansum((np.diff(log_growth_meas) - np.diff(log_growth_calc))**2, axis = None)
    elif version == 'new':
        for i, (meas, calc) in enumerate(zip(log_growth_meas, log_growth_calc)):
            s = np.zeros((len(meas), len(meas)))
            switching_times = np.array(sim.type.change_times)
            values = np.array([0] + list(sim.type.doses[i]))
            c_t = lambda t: values[np.searchsorted(switching_times, t, side='right')]
            for j in range(len(meas) - 1):
                t_l = sim.type.meas_times[j]
                t_r = sim.type.meas_times[j + 1]
                # WARNING: the following is a hacky and inaccurate way to integrate
                par_l = utils.get_fund_param_set(params, c_t(t_l)) 
                par_r = utils.get_fund_param_set(params, c_t(t_r))
                sigma_l = 0.8 * (par_l.b0 + par_r.d0) / calc[j]  # TODO: better approx for f0
                sigma_r = 0.8 * (par_r.b0 + par_r.d0) / calc[j + 1]  # TODO: better approx for f0
                dt = t_r - t_l 
                sigma = dt * (sigma_l + sigma_r) / 2 
                s[(j+1):,j] = sigma
            c = s @ s.T + (meas_err**2) * np.identity(len(meas))
            # WARNING: the following wont work with meas_error = 0
            delta = meas - calc
            inc = delta @ np.linalg.solve(c, delta)
            sum_of_sq += inc
    else:
        raise Exception("version not in [RMS-growth, RMS-growthrate, new]")

    end = time.time()
    elapsed = end - begin
    #if elapsed > 1:
    #    print(f'problematic params: {params}')
    return sum_of_sq

# a function to fit the parameters of a certain parameter regeme
# to a simulation or experiment. It uses the l-bfgs-b minimizer
# and does several basin hops to make sure it is not getting stuck 
# in a local minima
# it also takes in a strategy to deal with the initial f0. "equilib" allways assumes
# that it is equal to the equilibrium f0 for the parameters it is testing,
# given takes a given f0 (for example, the true value or 1) and "fit" allows one 
# to fit the value
def fit_params_log_growth_pulsed(sim, param_type, n_hops=3, meas_error=0.05, 
                                 liklihood_vers='RMS-growth', f0_strat="given", 
                                 f0_init=1):
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


