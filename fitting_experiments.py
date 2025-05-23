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

no_fits = 10
paralell = True
no_basin_hops = 4

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
    

def sum_of_sq_error_log_growth_pulsed(a, sim, f0_init = 10/11):
    begin = time.time()
    params = utils.LastYearParamSetLinearBD(*a)
    if np.any(np.array(a) > 1):
        #print(f'huge param {params}')
        return 1e100
    if np.any(np.array(a) * np.array([1] * 3 + [-1] + [1] * 5) < 0):
        #print('negative param {params}')
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


def fit_params_log_growth_pulsed(sim):
    obj = lambda a : \
            sum_of_sq_error_log_growth_pulsed(a, sim)
    bounds = scipy.optimize.Bounds([0.0] * 3 + [-0.1] + [0.0] * 5, 
                                   [0.1] * 3 + [0.0] + [0.1] * 5)

    x0 = [0.1] * 3 + [-0.1] + [0.1] * 5

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
        niter=no_basin_hops,  
        minimizer_kwargs=minimizer_kwargs,
        disp=True
    )
    x = result.x

    return utils.LastYearParamSetLinearBD(*x)

# Function partially written by ChatGPT
def plot_params_fit_log_growth_pulsed(sim, params):
    # Create figure
    fig = go.Figure()

    colors = px.colors.qualitative.Plotly

    log_growth_calc_true = np.log(utils.calc_meas_mat_bd(sim.type, lin_param_default, 10/11, 1).data)
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


def fit_one(_):
    sim = simulate_pulsed(lin_param_default, sim_type_pulsed, 1000, 100, 0.005)
    best_params = fit_params_log_growth_pulsed(sim)
    best_fit_error = sum_of_sq_error_log_growth_pulsed(list(best_params), sim)
    true_fit_error = sum_of_sq_error_log_growth_pulsed(list(used_params), sim)
    print(f'best fit error {best_fit_error}')
    print(f'true fit error {true_fit_error}')
    print(best_params)
    return (best_params, sim)

#param_est, sim = fit_one(None)
#plot_params_fit_log_growth_pulsed(sim, utils.LastYearParamSetLinearBD(*param_est))

def fit_one_tup(_):
    param_est, sim = fit_one(_)
    return (tuple(param_est), sim)


fits = []
sims = []

if paralell: # done by ChatGPT
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = list(executor.map(fit_one_tup, [None] * no_fits))
    fits = [utils.LastYearParamSetLinearBD(*param_est_tuple) for param_est_tuple, _ in results]
    sims = [sim for _, sim in results]
else:
    for i in range(no_fits):
        param_est, sim = fit_one(None)
        fits.append(param_est)
        sims.append(sim)

print(fits)

fit_ratios = []
for best_params, sim in zip(fits, sims):
    best_fit_error = sum_of_sq_error_log_growth_pulsed(list(best_params), sim)
    true_fit_error = sum_of_sq_error_log_growth_pulsed(list(used_params), sim)
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

def save_single_dataframe(df, index_file='param_est_index.txt', prefix='param_est_'):
    next_idx = get_next_index(prefix=prefix)
    filename = f'{prefix}{next_idx}.csv'
    df.to_csv(filename, index=False)

    # Metadata
    info = (
        f'File: {filename}\n'
        f'Time: {datetime.now().isoformat()}\n'
        f'Shape: {df.shape}\n'
        f'Columns: {list(df.columns)}\n'
        f'simulation type: {sim_type_pulsed}\n'
        f'fit ratios: {fit_ratios}\n'
        f'number of fits: {no_fits}\n'
        f'parameters: {used_params}\n'
        f'note: just a test\n'
        '---\n'
    )

    with open(index_file, 'a') as f:
        f.write(info)

    print(f"Saved: {filename}")
    print(f"Appended info to: {index_file}")

save_single_dataframe(df)
