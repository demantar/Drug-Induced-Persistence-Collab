import model_utils as utils
from simulate import *
import pandas as pd
import concurrent.futures
import os
import re
from datetime import datetime
from functools import partial
import pickle
from fitting import *

# a function that simulates an experiment of a certain type, fits
# parameters to the data, and prints info on how well the parameters
# were fit compared to the old parameters
def fit_one_tup(_, used_params, sim_type_pulsed, n_basin_hops, meas_sigma, true_cnt, 
                true_f0_init, liklihood_version="RMS-growth", f0_strat="true", deterministic_simp=False):
    n0 = round(true_cnt * true_f0_init)
    n1 = true_cnt - n0
    true_f0_init = n0 / true_cnt

    if deterministic_simp:
        sim = utils.calc_meas_mat(sim_type_pulsed, used_params, true_f0_init, [true_cnt] * len(sim_type_pulsed.doses), meas_sigma)
    else:
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

    best_fit_error = estimation_objective(best_params, sim, meas_err=meas_sigma, 
                                          version=liklihood_version, f0_init=true_f0_init)

    true_fit_error = estimation_objective(used_params, sim, meas_err=meas_sigma, 
                                          version=liklihood_version, f0_init=true_f0_init)

    print(f'best fit error {best_fit_error}')
    print(f'true fit error {true_fit_error}')
    print(best_params)
    return (tuple(best_params), f0_fit, sim)

# a function that simulates several experiments to better understand how
# estimateable the parameters are.
# note support for paralellization on multiple cores
def run_experiment_batch(used_params, sim_type_pulsed, n_experiments, true_cnt, 
                         true_f0_init, paralell = True, n_basin_hops = 3, 
                         meas_sigma = 0.05, liklihood_version='RMS-growth', 
                         f0_strat="true", deterministic_simp=False):
    bound_fit = partial(
        fit_one_tup,
        used_params=used_params,
        sim_type_pulsed=sim_type_pulsed,
        n_basin_hops=n_basin_hops,
        meas_sigma=meas_sigma,
        liklihood_version=liklihood_version,
        true_cnt=true_cnt,
        true_f0_init=true_f0_init,
        f0_strat=f0_strat,
        deterministic_simp=deterministic_simp
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
def run_and_save_experiment(used_params, sim_type_pulsed, n_experiments, 
                            true_cnt, true_f0_init, paralell = True, 
                            n_basin_hops = 3, file_pref = "param_est_", 
                            meas_sigma = 0.05, message = "no_message", 
                            liklihood_vers="RMS-growth", f0_strat="true",
                            deterministic_simp=False):
    fits, f0_fits, sims = run_experiment_batch(used_params, sim_type_pulsed, n_experiments, 
                                               true_cnt, true_f0_init, paralell, 
                                               n_basin_hops, meas_sigma, 
                                               liklihood_version=liklihood_vers, 
                                               f0_strat=f0_strat, 
                                               deterministic_simp=deterministic_simp)

    fit_ratios = []
    for best_params, sim in zip(fits, sims):
        best_fit_error = estimation_objective(best_params, sim, meas_err=meas_sigma, 
                                              version=liklihood_vers, f0_init=true_f0_init)
        true_fit_error = estimation_objective(used_params, sim, meas_err=meas_sigma, 
                                              version=liklihood_vers, f0_init=true_f0_init)
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
        "Parameter": "f0_init",
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
