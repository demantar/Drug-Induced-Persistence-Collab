## ----- This is a set of utilities for other scripts ------
# In this script, the two state model will be used. 
# Perhaps, I will make a generalization to a arbitrary state model later
from collections import namedtuple
import numpy as np
import scipy.integrate
import scipy.linalg

# 'datatype' for efficiently storing model parameters
# in the case where the parameters do not depend on the dose
# NOTE: it is important that the first argument of namedtuple exactly matches
# the name of the type
FundamentalParamSet = namedtuple('FundamentalParamSet', 'mu nu b0 d0 b1 d1')

# the following are parameter sets for various different parameter regimes.
# to make a new regime, just add a line and then add to the functions
# get_fund_param_set and get_bounds to explain how to get the fundamental 
# parameters from this regime and what are resonable bounds for the parameter

# Linear model from last summer article. Note that h_mu = k and h_nu = -m
LastYearParamSetLinear = namedtuple('LastYearParamSetLinear', 
                        'mu h_mu nu h_nu b0 d0 d_d0 b1 d1')
LastYearParamSetLinear_no_h_nu = namedtuple('LastYearParamSetLinear_no_h_nu', 
                        'mu h_mu nu b0 d0 d_d0 b1 d1')

# function that takes a prameter set par and a dose c and returns a fundamental
# parameter set for that dose
# in some places, they have been made positive and other formulas adjusted
# code should use numpy vectorized functions for c
def get_fund_param_set(par, c):
    if isinstance(par, LastYearParamSetLinear):
        return FundamentalParamSet(
                par.mu + par.h_mu * c, 
                np.maximum(par.nu + par.h_nu * c, 0), 
                par.b0, 
                par.d0 + par.d_d0 * c / (c + 1), 
                par.b1,
                par.d1
        )
    if isinstance(par, LastYearParamSetLinear_no_h_nu):
        return FundamentalParamSet(
                par.mu + par.h_mu * c, 
                np.maximum(par.nu, 0), 
                par.b0, 
                par.d0 + par.d_d0 * c / (c + 1), 
                par.b1,
                par.d1
        )

# a function that takes a parameter set type and returns bounds
# for the parameter. The fist list in the tuple is the lower bound
# and the latter is an upper bound.
def get_bounds(param_type):
    if param_type is LastYearParamSetLinear:
        return ([0.0] * 3 + [-0.1] + [0.0] * 5,
                [0.1] * 3 + [0.0] + [0.1] * 5)
    if param_type is LastYearParamSetLinear_no_h_nu:
        return ([0.0] * 8,
                [0.1] * 8)
    raise Exception("parameter regime does not have bounds")


# The following namedtuples are 'datatypes' to store types of experiments
# and measurements / simulations of those experiments

# Tuple for the type of a measurement / simulation
# at time meas_times[i] the doses become the i-th column in doses
# it is assumed that the dose at time zero is 0
# arrays should be numpy arrays
MeasurementType = namedtuple('MeasurementType', 
                                   'change_times meas_times doses')
# A tuple to store the results of a measurement / simulation
Measurement = namedtuple('Measurement', 'type data')

# A function that takes FixedParamSet and returns the infinatesimal generator matrix
def inf_gen_mat(par):
    lambda0 = par.b0 - par.d0
    lambda1 = par.b1 - par.d1
    return np.array([[lambda0 - par.mu, par.mu], 
                     [par.nu, lambda1 - par.nu]])

# A function that takes a measurment type and generates a percise measurement
# of the cell count assuming the deterministic simplification
# (the ODE) via matrix exponentiation
def calc_meas_mat(meas_type_pulsed, params, f0_init, n0):
    results = []
    for dose_sched in meas_type_pulsed.doses: # iterate through each tumor
        dose_result = []
        n = np.array([[f0_init, 1 - f0_init]]) * n0
        i_change = 0
        t_curr = 0
        # iterate through all measurement times
        for i_meas in range(len(meas_type_pulsed.meas_times)): 
            t_nxt = meas_type_pulsed.meas_times[i_meas]
            # a loop to update the dosage
            while (i_change < len(meas_type_pulsed.change_times) - 1
                   and meas_type_pulsed.change_times[i_change + 1] <= t_nxt):  
                par_fix = get_fund_param_set(params, dose_sched[i_change])
                t_new = meas_type_pulsed.change_times[i_change + 1]
                n = n @ scipy.linalg.expm((t_new - t_curr) * inf_gen_mat(par_fix)) 
                i_change += 1
                t_curr = t_new
            par_fix = get_fund_param_set(params, dose_sched[i_change])
            n = n @ scipy.linalg.expm((t_nxt - t_curr) * inf_gen_mat(par_fix)) 
            t_curr = t_nxt
            dose_result.append(np.sum(n, axis=None))
        results.append(dose_result)
    return Measurement(meas_type_pulsed, np.array(results))

# a function that returns the equilibrium f0 for a given fixed parameter set
def equilibf0(params):
    A = inf_gen_mat(params)
    # ChatGPT code to find dominant left eigenvector
    eigvals, eigvecs = np.linalg.eig(A.T)
    dominant_idx = np.argmax(eigvals) # TODO: shouold we have absolute value within?
    dominant_left_eigvec = eigvecs[:, dominant_idx]
    dominant_left_eigvec = dominant_left_eigvec[0] / np.sum(dominant_left_eigvec)
    return dominant_left_eigvec

            
# A function that takes in a FixedParamSet and returns the derivative of 
# f0
# *should* work with numpy arrays instead of numbers
def f0_prime(f0, fix_par):
    lambda0 = fix_par.b0 - fix_par.d0
    lambda1 = fix_par.b1 - fix_par.d1
    a = lambda1 - lambda0
    b = -(lambda1 - lambda0 + fix_par.mu + fix_par.nu)
    c = fix_par.nu
    return a * f0 * f0 + b * f0 + c

# A functinon that takes a FixedParamSet and a dosage and returns the derivative
# *should* work with numpy arrays instead of numbers
def f0_prime_c(f0, var_par, c):
    return f0_prime(f0, get_fund_param_set(var_par, c))

# A function that calculates the growth rate rho given f0 and 
# a fixed parameter set (denoted u in the prev article)
# *should* work with numpy arrays instead of numbers
def rho(f0, fix_par):
    lambda0 = fix_par.b0 - fix_par.d0
    lambda1 = fix_par.b1 - fix_par.d1
    return lambda0 * f0 + lambda1 * (1 - f0)

# similar to the previous function but given a variable parameter set and a dose
def rho_c(f0, var_par, c):
    return rho(f0, get_fund_param_set(var_par, c))

# Function that solves for f0 given a function c_t that takes in an
# numpy array of time and returns the doses at those times. It also takes in an 
# initial f_0 and a variable parameter and times to evaluate the solution at
def sol_f0(var_par, c_t, t_eval, f0_init):
    f0_ode = lambda t, y : f0_prime_c(y, var_par, c_t(t))
    return scipy.integrate.solve_ivp(f0_ode, (min(t_eval), max(t_eval)), 
                                     [f0_init], t_eval=t_eval).y.flatten()

# function that finds the logarithm of the relative total growth. That is 
# the integral of the growth rate. It is computed using simpsons rule
def log_growth(f0_t, var_par, c_t, t_eval):
    return scipy.integrate.cumulative_simpson(rho_c(f0_t, var_par, c_t(t_eval)), x=t_eval, initial=0)

