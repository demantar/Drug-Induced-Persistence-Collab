import model_utils as utils
from simulate import *
import scipy.optimize
import scipy.interpolate
import plotly.graph_objects as go
import plotly.express as px
import itertools
import pandas as pd
import multiprocessing as mp
import time
import skopt
import skopt.space

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
    h_nu = -0.00004,
    b0 = 0.04,
    d0 = 0.0,
    d_d0 = 0.08,
    b1 = 0.001,
    d1 = 0.0
)
    
"""
def sum_of_sq_error_log_growth_pulsed(a, sim, f0_init = 10/11):
    begin = time.time()
    params = utils.LastYearParamSetLinearBD(*a)
    if np.any(np.array(a) > 1):
        #print(f'huge param {params}')
        return 1e100
    if np.any(np.array(a) * np.array([1] * 3 + [-1] + [1] * 5) < 0):
        print('negative param {params}')
        return 1e100

    sum_of_sq = 0
    for counts, doses in zip(sim.data, sim.type.doses):
        log_growth_meas = np.log(counts / counts[0])
        doses_prepended = np.array([0] + list(doses))
        c_t = lambda t : doses_prepended[np.searchsorted(sim_type_pulsed.change_times, t, side="right")]
        solve_times = np.linspace(min(sim.type.meas_times), max(sim.type.meas_times), 100)
        f0 = utils.sol_f0_bd(params, c_t, solve_times, f0_init) # TODO: should f0_init be 1?

        if f0.size != 100:
            print("f0 shape wrong!")
            print(f'f0: {f0}')
            print(f'params: {params}')
            return 1e100

        log_growth_calc_detailed = utils.log_growth_bd(f0, params, c_t, solve_times)
        interp = scipy.interpolate.interp1d(solve_times, log_growth_calc_detailed)
        log_growth_calc = interp(sim.type.meas_times)

        sum_of_sq += np.sum((log_growth_meas - log_growth_calc)**2)

    end = time.time()
    elapsed = end - begin
    print(f'objective eval completed in {elapsed} seconds with fitnes {np.log(sum_of_sq)}')
    if elapsed > 1:
        print(f'problematic params: {params}')
    return sum_of_sq
"""

def sum_of_sq_error_log_growth_pulsed(a, sim, f0_init = 10/11):
    begin = time.time()
    params = utils.LastYearParamSetLinearBD(*a)
    if np.any(np.array(a) > 1):
        print(f'huge param {params}')
        return 1e100
    if np.any(np.array(a) * np.array([1] * 3 + [-1] + [1] * 5) < 0):
        print('negative param {params}')
        return 1e100

    log_growth_meas = np.log(sim.data)
    # WARNING hacky way to find n0
    log_growth_calc = np.log(utils.calc_meas_mat_bd(sim.type, params, f0_init, sim.data[0, 0]).data)
    #print(f'meas: {log_growth_meas}')
    #print(f'calc: {log_growth_calc}')
    
    sum_of_sq = np.sum((log_growth_meas - log_growth_calc)**2, axis = None)

    end = time.time()
    elapsed = end - begin
    #print(f'objective eval completed in {elapsed} seconds with fitnes {np.log(sum_of_sq)}')
    if elapsed > 1:
        print(f'problematic params: {params}')
    return sum_of_sq

def fit_params_log_growth_pulsed_incremental(sim):
    change_dom = lambda a : [np.exp(a[0]), 
                             np.exp(a[1]),
                             np.exp(a[2]), 
                             -np.exp(a[3]), 
                             np.exp(a[4]), 
                             np.exp(a[5]),
                             np.exp(a[6]),
                             np.exp(a[7]),
                             np.exp(a[8])]

    # CHAT begin
    """
    obj = lambda a : \
           np.log(sum_of_sq_error_log_growth_pulsed(change_dom(a), sim))
    #obj = lambda a : \
    #       sum_of_sq_error_log_growth_pulsed(a, sim)
    f = lambda x, y: obj([np.log(0.0004), 
                          -x, #np.log(0.0004), 
                          np.log(0.004), 
                          np.log(0.00004), 
                          np.log(0.04), 
                          -y, #np.log(0.00001), 
                          np.log(0.08), 
                          np.log(0.001), 
                          np.log(0.00001)])
    #f = lambda x, y: obj([0.0004, 
    #                      x, #0.0004, 
    #                      0.004, 
    #                      -0.00004, 
    #                      0.04, 
    #                      y, #0.00001, 
    #                      0.08, 
    #                      0.001, 
    #                      0.00001])
    f_vec = np.vectorize(f)
    x = np.linspace(0.0001, 0.999, 20)
    y = np.linspace(0.0001, 0.999, 20)
    X, Y = np.meshgrid(x, y)
    Z = f_vec(X, Y)
    Z[Z > 1e99] = None
    print(Z)

     #Create the surface plot
    fig = go.Figure(data=[go.Surface(z=Z, x=X, y=Y)])

     #Customize layout
    fig.update_layout(
        title='2D Function Plot: f(x, y) = sin(sqrt(x² + y²))',
        scene=dict(
            xaxis_title='x',
            yaxis_title='y',
            zaxis_title='f(x, y)'
        ),
        autosize=False,
        width=800,
        height=600
    )

    fig.show()
    print("plot done")
    # CHAT end
    """

    x = [-1.0] * 9
    #x = [0.1] * 3 + [-0.1] + [0.1] * 5

    #start = time.time()
    #obj(x)
    #end = time.time()
    #print(f'time for one eval{end - start}')
    
    pref_times = [5, 6, 7, 8, 9, 10]
    for i, i_t in enumerate(pref_times):
        if i_t == 0: # WARNING: does not work for only one measureDimensionment
            continue
        sim_pref = utils.Measurement(
                    utils.MeasurementTypePulsed(
                        change_times = sim.type.change_times,
                        meas_times = sim.type.meas_times[:i_t+1],
                        doses = sim.type.doses
                    ),
                    sim.data[:,:i_t+1], 
                )

        #print(sim_pref)


        obj = lambda a : \
                float(np.log(sum_of_sq_error_log_growth_pulsed(change_dom(a), sim_pref)))
        #obj = lambda a : \
        #        sum_of_sq_error_log_growth_pulsed(a, sim_pref)
        #if i == 0:
        res = scipy.optimize.minimize(obj, x, method='Nelder-Mead', tol=1e-8, options={'disp': True})
        x = res.x.tolist()
        #else:
            #res = skopt.gp_minimize(obj, [skopt.space.Real(-5, 0) for _ in range(9)], x0 = x.tolist(), verbose = True)
            #x = res.x
        print('=========================================================')
        print(f'prefix up to time {sim.type.meas_times[i_t]} with guess \
              {utils.LastYearParamSetLinearBD(*change_dom(x))} finished')
        print('=========================================================')


        #print(x)


    #res = gp_minimize(obj, 
    #                  dimensions=[(0.0, 0.1), (0.0, 0.1), (0.0, 0.1), 
    #                              (-0.1, 0.0), (0.0, 0.1), (0.0, 0.1),
    #                              (0.0, 0.1), (0.0, 0.1), (0.0, 0.1)],
    #                  n_calls=200,
    #                  acq_func="EI")
    #x = scipy.optimize.minimize(obj_normal_dom, [0.1] * 3 + [-0.1] + [0.1] * 5, tol=1e-6, constraints=constr, options={'disp': True}).x
    #x = scipy.optimize.minimize(obj, [-1] * 9, tol=1e-6, options={'disp': True}).x
    #x = scipy.optimize.differential_evolution(obj_normal_dom, [(0, 1)] * 3 + [(-1, 0)] + [(0, 1)] * 5, disp=True).x
    #x = scipy.optimize.dual_annealing(obj, [(-6, 1)] * 9, maxiter=2, callback=lambda x, f, c: print(x, f, c)).x
    return utils.LastYearParamSetLinearBD(*change_dom(x))
    #return utils.LastYearParamSetLinearBD(*x)

def fit_params_log_growth_pulsed_test(sim):
    change_dom = lambda a : [np.exp(a[0]), 
                             np.exp(a[1]),
                             np.exp(a[2]), 
                             -np.exp(a[3]), 
                             np.exp(a[4]), 
                             np.exp(a[5]),
                             np.exp(a[6]),
                             np.exp(a[7]),
                             np.exp(a[8])]

    x = [-1.0] * 9 + [3]

    initial_guess_ranges = [(0, 5), (0, 10)]
    initial_guesses = []
    for i, j in initial_guess_ranges:
        sim_pref = utils.Measurement(
                    utils.MeasurementTypePulsed(
                        change_times = sim.type.change_times,
                        meas_times = sim.type.meas_times[i:j+1],
                        doses = sim.type.doses
                    ),
                    sim.data[:,i:j+1], 
                )

        obj = lambda a : \
                np.log(sum_of_sq_error_log_growth_pulsed(change_dom(a[:-1]), sim_pref, 
                                                         1 / (1 + math.exp(a[-1]))))
        x = scipy.optimize.minimize(obj, x, method='Nelder-Mead', tol=1e-3, 
                                    options={'disp': True}).x
        initial_guesses.append(x[:-1])
        err = obj(x)
        print(f'guessrange ({i}, {j}) resulted in error {err}')

    # CHAT begin
    def weights(n, d):
        return np.random.dirichlet(np.ones(d), size=n)
    
    # CHAT end

    obj = lambda a : \
            np.log(sum_of_sq_error_log_growth_pulsed(change_dom(a), sim))
    best_fitness = 1e100
    best_guess = None
    for i in range(200):
        ws = weights(9, len(initial_guesses))
        vec = np.zeros(9)
        for guess, w in zip(initial_guesses, ws.T):
            #print(guess, w)
            #print(np.array(guess) * w)
            vec += np.array(guess) * w
        #print(f'linear_comb {vec}')
        fitness = obj(list(vec))
        if fitness < best_fitness:
            best_fitness = fitness
            best_guess = vec

    print(f'best_guess {best_guess} with error {best_fitness}')

    x = scipy.optimize.minimize(obj, best_guess, method='Nelder-Mead', tol=1e-3, 
                                options={'disp': True}).x
    print(x)
    print("hello")
    return utils.LastYearParamSetLinearBD(*change_dom(x))

            

        


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
    obj_normal_dom = lambda a : \
            sum_of_sq_error_log_growth_pulsed(a, sim)
    print(np.array([0.0] * 3 + [-np.inf] + [0.0] * 5).shape)
    print(np.identity(9).shape[0:1])
    constr = scipy.optimize.LinearConstraint(np.identity(9), 
                                             lb=np.array([0.0] * 3 + [-np.inf] + [0.0] * 5),
                                             ub=np.array([np.inf] * 3 + [0.0] + [np.inf] * 5),
                                             keep_feasible=np.repeat(True, 9))
    bounds = scipy.optimize.Bounds([0.0] * 3 + [-0.1] + [0.0] * 5, 
                                   [0.1] * 3 + [0.0] + [0.1] * 5)


    #res = gp_minimize(obj, 
    #                  dimensions=[(0.0, 0.1), (0.0, 0.1), (0.0, 0.1), 
    #                              (-0.1, 0.0), (0.0, 0.1), (0.0, 0.1),
    #                              (0.0, 0.1), (0.0, 0.1), (0.0, 0.1)],
    #                  n_calls=200,
    #                  acq_func="EI")
    #x = scipy.optimize.minimize(obj_normal_dom, [0.1] * 3 + [-0.1] + [0.1] * 5, tol=1e-6, constraints=constr, options={'disp': True}).x
    #x = scipy.optimize.minimize(obj, [-1] * 9, options={'disp': True}).x
    #x = scipy.optimize.basinhopping(obj_normal_dom, [0.1] * 3 + [-0.1] + [0.1] * 5, disp=True, minimizer_kwargs={"method": "L-BFGS-B", "bounds": bounds}).x
    x = scipy.optimize.minimize(obj_normal_dom, [0.1] * 3 + [-0.1] + [0.1] * 5, method='L-BFGS-B', bounds=bounds).x
    #x = scipy.optimize.differential_evolution(obj_normal_dom, [(0, 1)] * 3 + [(-1, 0)] + [(0, 1)] * 5, disp=True).x
    #x = scipy.optimize.dual_annealing(obj, [(-6, 1)] * 9, maxiter=2, callback=lambda x, f, c: print(x, f, c)).x
    #return utils.LastYearParamSetLinearBD(*change_dom(x))
    return utils.LastYearParamSetLinearBD(*x)

def plot_params_fit_log_growth_pulsed(sim, params):
    # Create figure
    fig = go.Figure()

    colors = px.colors.qualitative.Plotly

    log_growth_calc_true_new = np.log(utils.calc_meas_mat_bd(sim.type, lin_param_default, 10/11, 1).data)
    print(log_growth_calc_true_new)

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

        doses_prepended = np.array([0] + list(doses))
        c_t = lambda t : doses_prepended[np.searchsorted(sim_type_pulsed.change_times, t, side="right")]

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
            x=sim.type.meas_times, y=log_growth_calc,
            mode='lines',
            name=f'calc fit {i}',
            line=dict(color=color, dash='dot')  # Different style for g_i(t)
        ))


        # Add g_i(t)
        fig.add_trace(go.Scatter(
            x=sim.type.meas_times, y=log_growth_calc_true_new[i],
            mode='lines',
            name=f'calc true new {i}',
            line=dict(color=color, dash='dashdot')  # Different style for g_i(t)
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
    sim = simulate_pulsed(lin_param_default, sim_type_pulsed, 1000, 100, 0.000)
    print(sim)
    best_params = fit_params_log_growth_pulsed(sim)
    best_fit_error = sum_of_sq_error_log_growth_pulsed(list(best_params), sim)
    true_fit_error = sum_of_sq_error_log_growth_pulsed(list(lin_param_default), sim)
    print(f'best fit error {best_fit_error}')
    print(f'true fit error {true_fit_error}')
    print(best_params)
    return (tuple(best_params), sim)

param_est, sim = fit_one(None)
print(param_est)
print(sim)
plot_params_fit_log_growth_pulsed(sim, utils.LastYearParamSetLinearBD(*param_est))


no_fits = 0
def wrapper(_):
    return fit_one(_)[0]
with mp.Pool(processes=mp.cpu_count()) as pool:
    fits_tup = pool.map(wrapper, range(no_fits))


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

df.to_csv("fitting_data_2.csv")

print(df.to_string())
