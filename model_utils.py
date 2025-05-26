## ----- This is a set of utilities for other scripts ------
# In this script, the two state model will be used. 
# Perhaps, I will make a generalization to a arbitrary state model later
from collections import namedtuple
import numpy as np
import scipy.integrate
import scipy.linalg

# 'datatype' for efficiently storing model parameters
# in the case where the parameters do not depend on the dose
# *should* work with numpy arrays instead of numbers
FundamentalParamSet = namedtuple('FixedParamSet', 'mu nu lambda0 lambda1')
# Similar set with b0, d0, b1, d1 instead of lambda0, lambda1
FundamentalParamSetBD = namedtuple('FixedParamSet', 'mu nu b0 d0 b1 d1')

# the following are parameter sets for various different parameter regimes.
# to make a new regime, just add a line and then add to the function
# get_fund_param_set to explain how to get the fundamental parameters
# from this regime

# Linear model from last summer article. Note that h_mu = k and h_nu = -m
LastYearParamSetLinear = namedtuple('LastYearParamSetLinear', 
                        'mu h_mu nu h_nu lambda0 d_lambda0 lambda1')
LastYearParamSetLinearBD = namedtuple('LastYearParamSetLinear', 
                        'mu h_mu nu h_nu b0 d0 d_d0 b1 d1')

# Heaviside model from last summer article. Note changes in signs
LastYearParamSetHeaviside = namedtuple('LastYearParamSetHeaviside',
                            'mu d_mu nu d_nu lambda0 d_lambda0 lambda1')

# function that takes a VariableParamSet par and a dose c and returns a FixedParamSet
# for that dose, assuming Michaelis-Menten functions
# in some other places, they have been made positive and other formulas adjusted
# *should* work with numpy arrays instead of numbers
# code works better if numpy vectorized functions are used
def get_fund_param_set(par, c):
    if isinstance(par, LastYearParamSetLinear):
        return FundamentalParamSet(
                par.mu + par.h_mu * c, 
                par.nu + par.h_nu * c, 
                par.lambda0 + par.d_lambda0 * c / (c + 1),
                par.lambda1 
        )
    if isinstance(par, LastYearParamSetHeaviside):
        return FundamentalParamSet(
            par.mu if abs(c) <= 1e-9 else par.mu + par.d_mu,
            par.nu if abs(c) <= 1e-9 else par.nu,
            par.lambda0 * par.d_lambda0 * c / (c + 1),
            par.lambda1
        )

def get_fund_param_set_bd(par, c):
    if isinstance(par, LastYearParamSetLinearBD):
        return FundamentalParamSetBD(
                par.mu + par.h_mu * c, 
                np.maximum(par.nu + par.h_nu * c, 0), 
                par.b0, 
                par.d0 + par.d_d0 * c / (c + 1), 
                par.b1,
                par.d1
        )

def get_bounds(param_type):
    if param_type is LastYearParamSetLinearBD:
        return ([0.0] * 3 + [-0.1] + [0.0] * 5,
                [0.1] * 3 + [0.0] + [0.1] * 5)

# function that takes FixedParamSet and returns the infinatesimal generator matrix
def inf_gen_mat(par):
    return np.array([[par.lambda0 - par.mu, par.mu], 
                     [par.nu, par.lambda1 - par.nu]])
# for birth death model
def inf_gen_mat_bd(par):
    lambda0 = par.b0 - par.d0
    lambda1 = par.b1 - par.d1
    return np.array([[lambda0 - par.mu, par.mu], 
                     [par.nu, lambda1 - par.nu]])

def calc_meas_mat_bd(meas_type_pulsed, params, f0_init, n0):
    results = []
    for dose_sched in meas_type_pulsed.doses:
        dose_result = []
        n = np.array([[f0_init, 1 - f0_init]]) * n0
        i_change = 0
        t_curr = 0
        for i_meas in range(len(meas_type_pulsed.meas_times)):
            t_nxt = meas_type_pulsed.meas_times[i_meas]
            while (i_change < len(meas_type_pulsed.change_times) - 1
                   and meas_type_pulsed.change_times[i_change + 1] <= t_nxt):
                par_fix = get_fund_param_set_bd(params, dose_sched[i_change])
                t_new = meas_type_pulsed.change_times[i_change + 1]
                n = n @ scipy.linalg.expm((t_new - t_curr) * inf_gen_mat_bd(par_fix)) 
                i_change += 1
                t_curr = t_new
            par_fix = get_fund_param_set_bd(params, dose_sched[i_change])
            n = n @ scipy.linalg.expm((t_nxt - t_curr) * inf_gen_mat_bd(par_fix)) 
            t_curr = t_nxt
            dose_result.append(np.sum(n, axis=None))
        results.append(dose_result)
    return Measurement(meas_type_pulsed, np.array(results))

            



# A function that takes in a FixedParamSet and returns the derivative of 
# f0
# *should* work with numpy arrays instead of numbers
def f0_prime(f0, fix_par):
    a = fix_par.lambda1 - fix_par.lambda0
    b = -(fix_par.lambda1 - fix_par.lambda0 + fix_par.mu + fix_par.nu)
    c = fix_par.nu
    return a * f0 * f0 + b * f0 + c

# A functino that takes a FixedParamSet and a dosage and returns the derivative
# *should* work with numpy arrays instead of numbers
def f0_prime_c(f0, var_par, c):
    return f0_prime(f0, get_fund_param_set(var_par, c))

# A function that calculates the growth rate rho (denoted u in the prev article)
# *should* work with numpy arrays instead of numbers
def rho(f0, fix_par):
    return fix_par.lambda0 * f0 + fix_par.lambda1 * (1 - f0)

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
def log_growth(f0_t, var_par, c, t_eval):
    return scipy.integrate.cumulative_simpson(rho_c(f0_t, var_par, c), x=t_eval, initial=0)

## --- Dangerous copy pasted code adjusted for BD version 
# A function that takes in a FixedParamSet and returns the derivative of 
# f0
# *should* work with numpy arrays instead of numbers
def f0_prime_bd(f0, fix_par):
    lambda0 = fix_par.b0 - fix_par.d0
    lambda1 = fix_par.b1 - fix_par.d1
    a = lambda1 - lambda0
    b = -(lambda1 - lambda0 + fix_par.mu + fix_par.nu)
    c = fix_par.nu
    return a * f0 * f0 + b * f0 + c

# A functino that takes a FixedParamSet and a dosage and returns the derivative
# *should* work with numpy arrays instead of numbers
def f0_prime_c_bd(f0, var_par, c):
    return f0_prime_bd(f0, get_fund_param_set_bd(var_par, c))

# A function that calculates the growth rate rho (denoted u in the prev article)
# *should* work with numpy arrays instead of numbers
def rho_bd(f0, fix_par):
    lambda0 = fix_par.b0 - fix_par.d0
    lambda1 = fix_par.b1 - fix_par.d1
    return lambda0 * f0 + lambda1 * (1 - f0)

def rho_c_bd(f0, var_par, c):
    return rho_bd(f0, get_fund_param_set_bd(var_par, c))

# Function that solves for f0 given a function c_t that takes in an
# numpy array of time and returns the doses at those times. It also takes in an 
# initial f_0 and a variable parameter and times to evaluate the solution at
def sol_f0_bd(var_par, c_t, t_eval, f0_init):
    f0_ode = lambda t, y : f0_prime_c_bd(y, var_par, c_t(t))
    return scipy.integrate.solve_ivp(f0_ode, (min(t_eval), max(t_eval)), 
                                     [f0_init], t_eval=t_eval).y.flatten()

# function that finds the logarithm of the relative total growth. That is 
# the integral of the growth rate. It is computed using simpsons rule
def log_growth_bd(f0_t, var_par, c_t, t_eval):
    return scipy.integrate.cumulative_simpson(rho_c_bd(f0_t, var_par, c_t(t_eval)), x=t_eval, initial=0)

# A tuple to remember dimensions of measurments
MeasurementType = namedtuple('MeasurementType', 'doses times')
Measurement = namedtuple('Measurement', 'type data')

# Similar where the dose is varying in time
# at time meas_times[i] the doses become the i-th column in doses
# it is assumed that the dose at time zero is 0
MeasurementTypePulsed = namedtuple('MeasurementTypePulsed', 
                                   'change_times meas_times doses')

# convert MeasurementType to more general MeasurementTypePulsed for convenience
def meas_type_to_meas_type_pulsed(meas_type):
    return MeasurementTypePulsed(
            change_times = [0], 
            meas_times = meas_type.times,
            doses = np.array(meas_type.doses).reshape((len(meas_type.doses), 1))
    )
