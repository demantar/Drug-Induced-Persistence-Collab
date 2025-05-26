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

no_fits = 1
paralell = True
no_basin_hops = 2

test_sim_type = utils.MeasurementType(
        doses = [1,10,20,50,75,100],
        times = [0,10,20,30,40,50,60,70,80,90,100] 
)

sim_type_pulsed1 = utils.MeasurementTypePulsed(
    change_times = [0, 20, 40, 60, 80, 100],
    meas_times = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
    doses = np.array([[1, 0, 1, 0, 1, 0], 
                      [5, 0, 5, 0, 5, 0],
                      [10, 0, 10, 0, 10, 0],
                      [20, 0, 20, 0, 20, 0], 
                      [50, 0, 50, 0, 50, 0],
                      [75, 0, 75, 0, 75, 0],
                      [100, 0, 100, 0, 100, 0]])
)

sim_type_pulsed2 = utils.MeasurementTypePulsed(
    change_times = [0, 50],
    meas_times = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
    doses = np.array([[1, 0],
                      [5, 0],
                      [10, 0],
                      [20, 0],
                      [50, 0],
                      [75, 0],
                      [100, 0]])
)

sim_type_pulsed3 = utils.MeasurementTypePulsed(
    change_times = [0, 50],
    meas_times = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
    doses = np.array([[1, 0], [1, 0], 
                      [5, 0], [5, 0], 
                      [10, 0], [10, 0], 
                      [20, 0], [20, 0], 
                      [50, 0], [50, 0], 
                      [75, 0], [75, 0], 
                      [100, 0], [100, 0]])
)

sim_type_pulsed = sim_type_pulsed3

lin_param_default = utils.LastYearParamSetLinearBD(
    mu = 0.0004,
    h_mu = 0.00004,
    nu = 0.004,
    h_nu = -0.00004,
    b0 = 0.04,
    d0 = 0.0,
    d_d0 = 0.08,
    b1 = 0.001,
    d1 = 0.0
)

used_params = lin_param_default
used_param_type = type(used_params)
    
# A function that calculates the sum of square error for the parameter
# estimate params given a simulation or experiment sim
def sum_of_sq_error_log_growth_pulsed(params, sim, f0_init = 10/11):
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
    log_growth_calc = np.log(utils.calc_meas_mat_bd(sim.type, params, f0_init, sim.data[0, 0]).data)
    
    sum_of_sq = np.sum((log_growth_meas - log_growth_calc)**2, axis = None)

    end = time.time()
    elapsed = end - begin
    if elapsed > 1:
        print(f'problematic params: {params}')
    return sum_of_sq

# a function to fit the parameters of a certain parameter regeme
# to a simulation or experiment. It uses the l-bfgs-b minimizer
# and does several basin hops to make sure it is not getting stuck 
# in a local minima
def fit_params_log_growth_pulsed(sim, param_type, n_hops = 3):
    obj = lambda a : \
            sum_of_sq_error_log_growth_pulsed(param_type(*a), sim)

    #bounds = scipy.optimize.Bounds([0.0] * 3 + [-0.1] + [0.0] * 5, 
    #                               [0.1] * 3 + [0.0] + [0.1] * 5)
    lb, ub = utils.get_bounds(param_type)
    bounds = scipy.optimize.Bounds(lb, ub)

    #x0 = [0.1] * 3 + [-0.1] + [0.1] * 5

    x0 = np.random.uniform(low=lb, high=ub)

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

    return param_type(*x)

# Function partially written by ChatGPT
def plot_params_fit_log_growth_pulsed(sim, params, def_params=lin_param_default):
    # Create figure
    fig = go.Figure()

    colors = px.colors.qualitative.Plotly

    log_growth_calc_true = np.log(utils.calc_meas_mat_bd(sim.type, def_params, 10/11, 1).data)
    log_growth_calc_fitted = np.log(utils.calc_meas_mat_bd(sim.type, params, 10/11, 1).data)

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

def plot_params_fit_f0_pulsed(sim, params, def_params=lin_param_default): # TODO: change
    # Create figure
    fig = go.Figure()

    colors = px.colors.qualitative.Plotly

    log_growth_calc_true = np.log(utils.calc_meas_mat_bd(sim.type, def_params, 10/11, 1).data)
    log_growth_calc_fitted = np.log(utils.calc_meas_mat_bd(sim.type, params, 10/11, 1).data)

    for i, doses, counts in zip(itertools.count(), sim.type.doses, sim.data):
        color = colors[i % len(colors)]

        switching_times = np.array(sim.type.change_times)
        values = np.array([0] + list(doses))

        c_t = lambda t: values[np.searchsorted(switching_times, t, side='right')]
        f0 = utils.sol_f0_bd(params, c_t, sim.type.meas_times, 10/11)

        fig.add_trace(go.Scatter(
            x=sim.type.meas_times, y=f0,
            mode='lines',
            name=f'f_0 calc fit {i}',
            line=dict(color=color, dash='dash') 
        ))

        f0 = utils.sol_f0_bd(def_params, c_t, sim.type.meas_times, 10/11)

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
def fit_one_tup(_, used_params, sim_type_pulsed, n_basin_hops, meas_sigma):
    sim = simulate_pulsed(used_params, sim_type_pulsed, 1000, 100, meas_sigma)
    best_params = fit_params_log_growth_pulsed(sim, type(used_params), n_basin_hops)
    best_fit_error = sum_of_sq_error_log_growth_pulsed(best_params, sim)
    true_fit_error = sum_of_sq_error_log_growth_pulsed(used_params, sim)
    print(f'best fit error {best_fit_error}')
    print(f'true fit error {true_fit_error}')
    print(best_params)
    return (tuple(best_params), sim)

# a function that simulates several experiments to better understand how
# estimateable the parameters are
def run_experiment_batch(used_params, sim_type_pulsed, n_fits, paralell = True, n_basin_hops = 3, meas_sigma = 0.05):
    bound_fit = partial(
        fit_one_tup,
        used_params=used_params,
        sim_type_pulsed=sim_type_pulsed,
        n_basin_hops=n_basin_hops,
        meas_sigma=meas_sigma
    )

    fits = []
    sims = []

    if paralell: # done by ChatGPT
        with concurrent.futures.ProcessPoolExecutor() as executor:
            results = list(executor.map(bound_fit, [None] * n_fits))
        fits = [used_param_type(*param_est_tuple) for param_est_tuple, _ in results]
        sims = [sim for _, sim in results]
    else:
        for i in range(n_fits):
            param_est, sim = bound_fit(None)
            fits.append(param_est)
            sims.append(sim)
    
    return fits, sims

# a function that calls run_experiment_batch and saves the results in 
# such a way that it is easy to find and work with them
def run_and_save_experiment(used_params, sim_type_pulsed, n_fits, paralell = True, n_basin_hops = 3, file_pref = "param_est_", meas_sigma = 0.05, message = "no_message"):
    fits, sims = run_experiment_batch(used_params, sim_type_pulsed, n_fits, paralell, n_basin_hops, meas_sigma)

    fit_ratios = []
    for best_params, sim in zip(fits, sims):
        best_fit_error = sum_of_sq_error_log_growth_pulsed(best_params, sim)
        true_fit_error = sum_of_sq_error_log_growth_pulsed(used_params, sim)
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

    # Add true values
    for param in lin_param_default._fields:
        data.append({
            "Parameter": param,
            "Value": getattr(lin_param_default, param),
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
            f'Fit-ratios: {fit_ratios}\n'
            f'Number of Fits: {no_fits}\n'
            f'Parameters: {used_params}\n'
            f'Message: {message}\n'
            '---\n'
        )

        with open(index_file, 'a') as f:
            f.write(info)

        print(f"Saved: {filename_df}")
        print(f"Saved: {filename_data}")
        print(f"Appended info to: {index_file}")

    save_single(df, fits, sims)

#run_and_save_experiment(used_params, sim_type_pulsed, no_fits, paralell = True, n_basin_hops = 3, file_pref = "param_est_")
