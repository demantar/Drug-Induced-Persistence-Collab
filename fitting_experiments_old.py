import model_utils as utils
from simulate import *
import scipy.optimize
import scipy.interpolate
import plotly.graph_objects as go
import plotly.express as px
import itertools
import pandas as pd
import multiprocessing as mp

test_sim_type = utils.MeasurementType(
        doses = [1,10,20,50,75,100],
        times = [0,10,20,30,40,50,60,70,80,90,100] 
)

sim_type_pulsed = utils.MeasurementTypePulsed(
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

lin_param_default = utils.LastYearParamSetLinearBD(
    mu = 0.0004,
    h_mu = 0.0004,
    nu = 0.004,
    h_nu = -0.0004,
    b0 = 0.04,
    d0 = 0.0,
    d_d0 = 0.08,
    b1 = 0.001,
    d1 = 0.0
)

def sum_of_sq_error_rho(mu, h_mu, nu, h_nu, b0, d0, d_d0, b1, d1, sim_type, rhos, eval_t):
    params = utils.LastYearParamSetLinearBD(mu, h_mu, nu, h_nu, b0, 
                                            d0, d_d0, b1, d1)
    sum_of_sq = 0
    for rho_meas, c in zip(rhos, sim_type.doses):
        print(rho_meas.shape)
        c_t = lambda t : c
        f0 = utils.sol_f0_bd(params, c_t, eval_t, 1) # TODO: should f0_init be 1?
        # NOTE: should eval_t be finer in this function all?
        rho_calc = utils.rho_c_bd(f0, params, c)
        sum_of_sq += np.sum((rho_meas - rho_calc)**2)

    return sum_of_sq

# function that takes the array of rho arrays from utils.rho_from_meas
# and returns the best fit parameters according to leas squares 
# WARNING: the differentiation of the time series seems to be biased at the moment
def fit_params_rho(sim_type, rhos, eval_t):
    obj = lambda mu, h_mu, nu, h_nu, b0, d0, d_d0, b1, d1 : \
            sum_of_sq_error_rho(mu, h_mu, nu, h_nu, b0, d0, d_d0, b1, d1, 
                            sim_type, rhos, eval_t)

#no_fits = 1
#for i in range(no_fits):
#    sim = simulateFixedDoses(lin_param_default, test_sim_type, 1000, 100)
#    rhos = utils.rho_from_meas(sim, test_sim_type.times, 0)
#    fit_params(test_sim_type, rhos, test_sim_type.times) 
    
def sum_of_sq_error_log_growth(a, sim):
    params = utils.LastYearParamSetLinearBD(*a)

    sum_of_sq = 0
    for counts, c in zip(sim.data, sim.type.doses):
        log_growth_meas = np.log(counts / counts[0])
        c_t = lambda t : c

        solve_times = np.linspace(min(sim.type.times), max(sim.type.times), 100)
        f0 = utils.sol_f0_bd(params, c_t, solve_times, 10/11) # TODO: should f0_init be 1?

        if f0.size != 100:
            print(f'Weird shape: {f0.shape}')
        # NOTE: should eval_t be finer in this function all?
        log_growth_calc_detailed = utils.log_growth_bd(f0, params, c, solve_times)
        interp = scipy.interpolate.interp1d(solve_times, log_growth_calc_detailed)
        log_growth_calc = interp(sim.type.times)

        sum_of_sq += np.sum((log_growth_meas - log_growth_calc)**2)

    return sum_of_sq

def fit_params_log_growth(sim):
    change_dom = lambda a : [math.exp(a[0]), 
                             math.exp(a[1]),
                             math.exp(a[2]), 
                             -math.exp(a[3]), 
                             math.exp(a[4]), 
                             math.exp(a[5]),
                             math.exp(a[6]),
                             math.exp(a[7]),
                             math.exp(a[8])]
    obj = lambda a : \
            sum_of_sq_error_log_growth(change_dom(a), sim)
    #res = gp_minimize(obj, 
    #                  dimensions=[(0.0, 0.1), (0.0, 0.1), (0.0, 0.1), 
    #                              (-0.1, 0.0), (0.0, 0.1), (0.0, 0.1),
    #                              (0.0, 0.1), (0.0, 0.1), (0.0, 0.1)],
    #                  n_calls=200,
    #                  acq_func="EI")
    x = scipy.optimize.minimize(obj, [-1] * 9, tol=1e-1, options={'disp': True}).x
    return utils.LastYearParamSetLinearBD(*change_dom(x))

def plot_params_fit_log_growth(sim, params):
    # Create figure
    fig = go.Figure()

    colors = px.colors.qualitative.Plotly

    # Example: 10 pairs of functions
    for i, c, counts in zip(itertools.count(), sim.type.doses, sim.data):

        color = colors[i % len(colors)]

        # Add simulated trace
        fig.add_trace(go.Scatter(
            x=sim.type.times, y=np.log(counts/counts[0]),
            mode='lines',
            name=f'sim {c}',
            line=dict(color=color, dash='solid')
        ))

        c_t = lambda t : c

        solve_times = np.linspace(min(sim.type.times), max(sim.type.times), 100)
        f0 = utils.sol_f0_bd(lin_param_default, c_t, solve_times, 10/11) # TODO: should f0_init be 1?
        # NOTE: should eval_t be finer in this function all?
        log_growth_calc_detailed = utils.log_growth_bd(f0, lin_param_default, c, solve_times)
        interp = scipy.interpolate.interp1d(solve_times, log_growth_calc_detailed)
        log_growth_calc = interp(sim.type.times)

        # Add g_i(t)
        fig.add_trace(go.Scatter(
            x=solve_times, y=log_growth_calc_detailed,
            mode='lines',
            name=f'calc true {c}',
            line=dict(color=color, dash='dash')  # Different style for g_i(t)
        ))
        
        solve_times = np.linspace(min(sim.type.times), max(sim.type.times), 100)
        f0 = utils.sol_f0_bd(params, c_t, solve_times, 10/11) # TODO: should f0_init be 1?
        # NOTE: should eval_t be finer in this function all?
        log_growth_calc_detailed = utils.log_growth_bd(f0, params, c, solve_times)
        interp = scipy.interpolate.interp1d(solve_times, log_growth_calc_detailed)
        log_growth_calc = interp(sim.type.times)

        # Add g_i(t)
        fig.add_trace(go.Scatter(
            x=sim.type.times, y=log_growth_calc,
            mode='lines',
            name=f'calc fit {c}',
            line=dict(color=color, dash='dot')  # Different style for g_i(t)
        ))

    # Update layout
    fig.update_layout(
        title='10 Pairs of Functions of Time',
        xaxis_title='Time',
        yaxis_title='Function Value',
        legend_title='Functions',
        template='plotly_dark',  # Optional styling
        height=600
    )

    fig.show()

#no_fits = 1
#for i in range(no_fits):
#    sim = simulateFixedDoses(lin_param_default, test_sim_type, 1000, 100)
#    rhos = utils.rho_from_meas(sim, test_sim_type.times, 0)
#    fit_params(test_sim_type, rhos, test_sim_type.times) 
    
def sum_of_sq_error_log_growth_pulsed(a, sim):
    params = utils.LastYearParamSetLinearBD(*a)

    sum_of_sq = 0
    for counts, doses in zip(sim.data, sim.type.doses):
        log_growth_meas = np.log(counts / counts[0])
        c_t = lambda t : doses[np.searchsorted(sim_type_pulsed.change_times, t, side="right")]
        solve_times = np.linspace(min(sim.type.meas_times), max(sim.type.meas_times), 100)
        f0 = utils.sol_f0_bd(params, c_t, solve_times, 10/11) # TODO: should f0_init be 1?

        log_growth_calc_detailed = utils.log_growth_bd(f0, params, c_t, solve_times)
        interp = scipy.interpolate.interp1d(solve_times, log_growth_calc_detailed)
        log_growth_calc = interp(sim.type.meas_times)

        sum_of_sq += np.sum((log_growth_meas - log_growth_calc)**2)

    return sum_of_sq

def fit_params_log_growth_pulsed(sim):
    change_dom = lambda a : [math.exp(a[0]), 
                             math.exp(a[1]),
                             math.exp(a[2]), 
                             -math.exp(a[3]), 
                             math.exp(a[4]), 
                             math.exp(a[5]),
                             math.exp(a[6]),
                             math.exp(a[7]),
                             math.exp(a[8])]
    obj = lambda a : \
            sum_of_sq_error_log_growth_pulsed(change_dom(a), sim)
    #res = gp_minimize(obj, 
    #                  dimensions=[(0.0, 0.1), (0.0, 0.1), (0.0, 0.1), 
    #                              (-0.1, 0.0), (0.0, 0.1), (0.0, 0.1),
    #                              (0.0, 0.1), (0.0, 0.1), (0.0, 0.1)],
    #                  n_calls=200,
    #                  acq_func="EI")
    x = scipy.optimize.minimize(obj, [-1] * 9, tol=1e-1, options={'disp': True}).x
    return utils.LastYearParamSetLinearBD(*change_dom(x))

def plot_params_fit_log_growth_pulsed(sim, params):
    # Create figure
    fig = go.Figure()

    colors = px.colors.qualitative.Plotly

    # Example: 10 pairs of functions
    for i, doses, counts in zip(itertools.count(), sim.type.doses, sim.data):

        color = colors[i % len(colors)]

        # Add simulated trace
        fig.add_trace(go.Scatter(
            x=sim.type.meas_times, y=np.log(counts/counts[0]),
            mode='lines',
            name=f'sim {i}',
            line=dict(color=color, dash='solid')
        ))

        c_t = lambda t : doses[np.searchsorted(sim_type_pulsed.change_times, t, side="right")]

        solve_times = np.linspace(min(sim.type.meas_times), max(sim.type.meas_times), 100)
        f0 = utils.sol_f0_bd(lin_param_default, c_t, solve_times, 10/11) # TODO: should f0_init be 1?
        # NOTE: should eval_t be finer in this function all?
        log_growth_calc_detailed = utils.log_growth_bd(f0, lin_param_default, c_t, solve_times)
        interp = scipy.interpolate.interp1d(solve_times, log_growth_calc_detailed)
        log_growth_calc = interp(sim.type.meas_times)

        # Add g_i(t)
        fig.add_trace(go.Scatter(
            x=solve_times, y=log_growth_calc_detailed,
            mode='lines',
            name=f'calc true {i}',
            line=dict(color=color, dash='dash')  # Different style for g_i(t)
        ))
        
        solve_times = np.linspace(min(sim.type.meas_times), max(sim.type.meas_times), 100)
        f0 = utils.sol_f0_bd(params, c_t, solve_times, 10/11) # TODO: should f0_init be 1?
        # NOTE: should eval_t be finer in this function all?
        log_growth_calc_detailed = utils.log_growth_bd(f0, params, c_t, solve_times)
        interp = scipy.interpolate.interp1d(solve_times, log_growth_calc_detailed)
        log_growth_calc = interp(sim.type.meas_times)

        # Add g_i(t)
        fig.add_trace(go.Scatter(
            x=sim.type.times, y=log_growth_calc,
            mode='lines',
            name=f'calc fit {i}',
            line=dict(color=color, dash='dot')  # Different style for g_i(t)
        ))

    # Update layout
    fig.update_layout(
        title='10 Pairs of Functions of Time',
        xaxis_title='Time',
        yaxis_title='Function Value',
        legend_title='Functions',
        template='plotly_dark',  # Optional styling
        height=600
    )

    fig.show()


#no_fits = 10
#fits = []
#for i in range(no_fits):
#    sim = simulateFixedDoses(lin_param_default, test_sim_type, 1000, 100, 0.005)
#    best_params = fit_params_log_growth(sim) 
#    print(f'best fit error {sum_of_sq_error_log_growth(list(best_params), sim)}')
#    print(f'true fit error {sum_of_sq_error_log_growth(list(lin_param_default), sim)}')
#    print(best_params)
#    fits.append(best_params)

def fit_one(_):
    sim = simulateFixedDoses(lin_param_default, test_sim_type, 1000, 100, 0.005)
    print(sim)
    best_params = fit_params_log_growth(sim)
    best_fit_error = sum_of_sq_error_log_growth(list(best_params), sim)
    true_fit_error = sum_of_sq_error_log_growth(list(lin_param_default), sim)
    print(f'best fit error {best_fit_error}')
    print(f'true fit error {true_fit_error}')
    print(best_params)
    return (tuple(best_params), sim)

param_est, sim = fit_one(None)
print(param_est)
print(sim)
plot_params_fit_log_growth(sim, utils.LastYearParamSetLinearBD(*param_est))


no_fits = 0
with mp.Pool(processes=mp.cpu_count()) as pool:
    fits_tup = pool.map(lambda _ : fit_one(_)[0], range(no_fits))


fits = list(map(lambda t : utils.LastYearParamSetLinearBD(*t), fits_tup))

print(fits)


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

#df.to_csv("fitting_data.csv")

print(df.to_string())

#fig = px.strip(
#    df,
#    x="Value",
#    y="Parameter",
#    color="Type",
#    stripmode="overlay",
#    hover_data=["Run"],
#    orientation="v"
#)

fig = px.box(
    df,
    x="Parameter",
    y="Value",
    points="all",
    color="Type"
)

# Log x-axis and style
fig.update_yaxes(type="log")
fig.update_layout(title="Parameter Estimates vs True Values", height=400)

fig.show()




