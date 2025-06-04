# functions for fitting parameters for Measurement instances and 
# testing the effeciveness of fits via simulations
# TODO: split this in two
import model_utils as utils
from simulate import *
import scipy.optimize
import scipy.interpolate
import plotly.graph_objects as go
import plotly.express as px
import itertools
import pandas as pd
import concurrent.futures
import time
import os
import re
from datetime import datetime
from functools import partial
import pickle
import numpy as np

    
# A function that calculates the objective to minimize in the parameter estimation
def estimation_objective(params, sim, f0_init = 10/11, version='RMS-growth', meas_err=0.05):
    begin = time.time()
    a = list(params)
    #params = utils.LastYearParamSetLinearBD(*a)
    if np.any(np.array(a) > 1):
        return 1e100
    lb, ub = utils.get_bounds(type(params))
    if np.any((a < lb) | (a > ub)):
        return 1e100

    log_growth_meas = np.log(sim.data)
    # WARNING hacky way to find n0
    log_growth_calc = np.log(utils.calc_meas_mat(sim.type, params, f0_init, sim.data[0, 0]).data)
    
    sum_of_sq = 0
    if version == 'RMS-growth':
        sum_of_sq = np.sum((log_growth_meas - log_growth_calc)**2, axis = None)
    elif version == 'RMS-growthrate':
        sum_of_sq = np.sum((np.diff(log_growth_meas) - np.diff(log_growth_calc))**2, axis = None)
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
def fit_params_log_growth_pulsed(sim, param_type, n_hops=3, meas_error=0.05, liklihood_vers='RMS-growth', f0_strat="given", f0_init=10/11):
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

# function to plot the logarithm of the size of the simulation,
# the deterministic simplification given true parameter def_params, 
# and the deterministic simplficiation given param estimate params
# It is good for testing and debugging parameter estimation
# Function partially written by ChatGPT
def plot_params_fit_log_growth_pulsed(sim, params, def_params):
    # Create figure
    fig = go.Figure()

    colors = px.colors.qualitative.Plotly

    log_growth_calc_true = np.log(utils.calc_meas_mat(sim.type, def_params, 10/11, 1).data)
    log_growth_calc_fitted = np.log(utils.calc_meas_mat(sim.type, params, 10/11, 1).data)

    for i, doses, counts in zip(itertools.count(), sim.type.doses, sim.data):

        color = colors[i % len(colors)]

        fig.add_trace(go.Scatter(
            x=sim.type.meas_times, y=np.log(counts/counts[0]),
            mode='lines',
            name=f'sim {i}',
            line=dict(color=color, dash='solid')
        ))

        fig.add_trace(go.Scatter(
            x=sim.type.meas_times, y=log_growth_calc_fitted[i],
            mode='lines',
            name=f'calc fitted {i}',
            line=dict(color=color, dash='dash')  
        ))

        fig.add_trace(go.Scatter(
            x=sim.type.meas_times, y=log_growth_calc_true[i],
            mode='lines',
            name=f'calc true {i}',
            line=dict(color=color, dash='dashdot') 
        ))

    # Update layout
    fig.update_layout(
        title='10 Pairs of Functions of Time',
        xaxis_title='Time',
        yaxis_title='Function Value',
        legend_title='Functions',
        template='plotly_dark', 
        height=600
    )

    fig.show()

# similar to last function but plots difference
def plot_params_fit_log_growth_pulsed_diff(sim, params, def_params):
    # Create figure
    fig = go.Figure()

    colors = px.colors.qualitative.Plotly

    log_growth_calc_true = np.log(utils.calc_meas_mat(sim.type, def_params, 10/11, 1).data)
    log_growth_calc_fitted = np.log(utils.calc_meas_mat(sim.type, params, 10/11, 1).data)

    for i, doses, counts in zip(itertools.count(), sim.type.doses, sim.data):

        color = colors[i % len(colors)]

        fig.add_trace(go.Scatter(
            x=sim.type.meas_times, y=np.log(counts/counts[0]) - log_growth_calc_true[i],
            mode='lines',
            name=f'sim - true {i}',
            line=dict(color=color, dash='solid')
        ))

        fig.add_trace(go.Scatter(
            x=sim.type.meas_times, y=log_growth_calc_fitted[i] - log_growth_calc_true[i],
            mode='lines',
            name=f'calc fitted - calc_true {i}',
            line=dict(color=color, dash='dash')  
        ))

    # Update layout
    fig.update_layout(
        title='10 Pairs of Functions of Time',
        xaxis_title='Time',
        yaxis_title='Function Value',
        legend_title='Functions',
        template='plotly_dark', 
        height=600
    )

    fig.show()

# another testing / debugging function to test how the estimate for f0
# calculated from the estimated parameters compares to the deterministic
# estimation of f0 given the true parameters
def plot_params_fit_f0_pulsed(sim, params, def_params): 
    # Create figure
    fig = go.Figure()

    colors = px.colors.qualitative.Plotly

    log_growth_calc_true = np.log(utils.calc_meas_mat(sim.type, def_params, 10/11, 1).data)
    log_growth_calc_fitted = np.log(utils.calc_meas_mat(sim.type, params, 10/11, 1).data)

    for i, doses, counts in zip(itertools.count(), sim.type.doses, sim.data):
        color = colors[i % len(colors)]

        switching_times = np.array(sim.type.change_times)
        values = np.array([0] + list(doses))

        c_t = lambda t: values[np.searchsorted(switching_times, t, side='right')]
        f0 = utils.sol_f0(params, c_t, sim.type.meas_times, 10/11)

        fig.add_trace(go.Scatter(
            x=sim.type.meas_times, y=f0,
            mode='lines',
            name=f'f_0 calc fit {i}',
            line=dict(color=color, dash='dash') 
        ))

        f0 = utils.sol_f0(def_params, c_t, sim.type.meas_times, 10/11)

        fig.add_trace(go.Scatter(
            x=sim.type.meas_times, y=f0,
            mode='lines',
            name=f'f_0 calc def {i}',
            line=dict(color=color, dash='dot') 
        ))

    # Update layout
    fig.update_layout(
        title='10 Pairs of Functions of Time',
        xaxis_title='Time',
        yaxis_title='Function Value',
        legend_title='Functions',
        template='plotly_dark', 
        height=600
    )

    print("hello")
    fig.show()

# a function that simulates an experiment of a certain type, fits
# parameters to the data, and prints info on how well the parameters
# were fit compared to the old parameters
def fit_one_tup(_, used_params, sim_type_pulsed, n_basin_hops, meas_sigma, true_cnt, 
                true_f0_init, liklihood_version="RMS-growth", f0_strat="true"):
    n0 = round(true_cnt * true_f0_init)
    n1 = true_cnt - n0
    true_f0_init = n0 / true_cnt
    sim = simulate(used_params, sim_type_pulsed, n0, n1, meas_sigma)
    if f0_strat == "true":
        best_params, f0_fit = fit_params_log_growth_pulsed(sim, type(used_params), n_basin_hops, 
                                                   liklihood_vers=liklihood_version,
                                                   meas_error=meas_sigma, f0_strat="given", 
                                                   f0_init=true_f0_init)
    elif f0_strat == "=1":
        best_params, f0_fit = fit_params_log_growth_pulsed(sim, type(used_params), n_basin_hops, 
                                                   liklihood_vers=liklihood_version,
                                                   meas_error=meas_sigma, f0_strat="given", 
                                                   f0_init=1)
    elif f0_strat in ["equilib", "fit"]:
        best_params, f0_fit = fit_params_log_growth_pulsed(sim, type(used_params), n_basin_hops, 
                                                   liklihood_vers=liklihood_version,
                                                   meas_error=meas_sigma, f0_strat=f0_strat)
    else:
        raise Exception("f0_strat not in [true, equilib, =1, fit]")

    best_fit_error = estimation_objective(best_params, sim, meas_err=meas_sigma, version=liklihood_version)
    true_fit_error = estimation_objective(used_params, sim, meas_err=meas_sigma, version=liklihood_version)
    print(f'best fit error {best_fit_error}')
    print(f'true fit error {true_fit_error}')
    print(best_params)
    return (tuple(best_params), f0_fit, sim)

# a function that simulates several experiments to better understand how
# estimateable the parameters are.
# note support for paralellization on multiple cores
def run_experiment_batch(used_params, sim_type_pulsed, n_experiments, true_cnt, true_f0_init, paralell = True, n_basin_hops = 3, meas_sigma = 0.05, liklihood_version='RMS-growth', f0_strat="true"):
    bound_fit = partial(
        fit_one_tup,
        used_params=used_params,
        sim_type_pulsed=sim_type_pulsed,
        n_basin_hops=n_basin_hops,
        meas_sigma=meas_sigma,
        liklihood_version=liklihood_version,
        true_cnt=true_cnt,
        true_f0_init=true_f0_init,
        f0_strat=f0_strat
    )

    fits = []
    f0_fits = []
    sims = []

    used_param_type = type(used_params)
    if paralell: # done by ChatGPT
        with concurrent.futures.ProcessPoolExecutor() as executor:
            results = list(executor.map(bound_fit, [None] * n_experiments))
        fits = [used_param_type(*param_est_tuple) for param_est_tuple, _, _ in results]
        sims = [sim for _, _, sim in results]
        f0_fits = [f0_fit for _, f0_fit, _ in results]
    else:
        for i in range(n_experiments):
            param_est, f0_fit, sim = bound_fit(None)
            fits.append(param_est)
            f0_fits.append(f0_fit)
            sims.append(sim)
    
    return fits, f0_fits, sims

# a function that calls run_experiment_batch and saves the results in 
# such a way that it is easy to find and work with them
def run_and_save_experiment(used_params, sim_type_pulsed, n_experiments, true_cnt, true_f0_init, paralell = True, n_basin_hops = 3, file_pref = "param_est_", meas_sigma = 0.05, message = "no_message", liklihood_vers="RMS-growth", f0_strat="true"):
    print(f'rase true_f0_init {true_f0_init}')
    fits, f0_fits, sims = run_experiment_batch(used_params, sim_type_pulsed, n_experiments, true_cnt, true_f0_init, paralell, n_basin_hops, meas_sigma, liklihood_version=liklihood_vers, f0_strat=f0_strat)

    fit_ratios = []
    for best_params, sim in zip(fits, sims):
        best_fit_error = estimation_objective(best_params, sim, meas_err=meas_sigma, version=liklihood_vers)
        true_fit_error = estimation_objective(used_params, sim, meas_err=meas_sigma, version=liklihood_vers)
        fit_ratios.append(best_fit_error / true_fit_error)


    # Flatten into a long-format DataFrame
    data = []
    for i, est in enumerate(fits):
        for param in est._fields:
            data.append({
                "Parameter": param,
                "Value": getattr(est, param),
                "Run": i,
                "Type": "Estimation"
            })

    for i, f0_fit in enumerate(f0_fits):
        data.append({
            "Parameter": "f0_init",
            "Value": f0_fit,
            "Run": i,
            "Type": "Estimation"
        })

    # Add true values
    for param in used_params._fields:
        data.append({
            "Parameter": param,
            "Value": getattr(used_params, param),
            "Run": -1,
            "Type": "True"
        })

    data.append({
        "Parameter": "f0",
        "Value": true_f0_init,
        "Run": -1,
        "Type": "True"
    })

    df = pd.DataFrame(data)

    def get_next_index(prefix='param_est_', suffix='.csv'):
        existing = [f for f in os.listdir() if f.startswith(prefix) and f.endswith(suffix)]
        indices = []
        for fname in existing:
            match = re.match(rf'{re.escape(prefix)}(\d+){re.escape(suffix)}', fname)
            if match:
                indices.append(int(match.group(1)))
        return max(indices, default=0) + 1

    def save_single(df, fits, sims, index_file='param_est_index.txt'):
        next_idx = get_next_index(prefix=file_pref)
        filename_df = f'{file_pref}{next_idx}.csv'
        df.to_csv(filename_df, index=False)

        next_idx = get_next_index(prefix='data_', suffix='.pkl')
        filename_data = f'data_{next_idx}.pkl'
        with open(filename_data, 'wb') as f:
            pickle.dump((fits, sims, used_params), f)

        # Metadata
        info = (
            f'Dataframe File: {filename_df}\n'
            f'Pickle Data File: {filename_data}\n'
            f'Time: {datetime.now().isoformat()}\n'
            f'Shape: {df.shape}\n'
            f'Columns: {list(df.columns)}\n'
            f'Simulation Type: {sim_type_pulsed}\n'
            f'Liklihood version: {liklihood_vers}\n'
            f'Fit-ratios: {fit_ratios}\n'
            f'Number of experiments: {n_experiments}\n'
            f'Parameters: {used_params}\n'
            f'Initial f0 strategy: {f0_strat}'
            f'Initial cell count: {true_cnt}'
            f'Message: {message}\n'
            '---\n'
        )

        with open(index_file, 'a') as f:
            f.write(info)

        print(f"Saved: {filename_df}")
        print(f"Saved: {filename_data}")
        print(f"Appended info to: {index_file}")

    save_single(df, fits, sims)
