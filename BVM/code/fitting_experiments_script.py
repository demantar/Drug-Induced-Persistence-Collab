# A file for queuing and saving simulated experiments
import fitting_experiments
import model_utils as utils
import numpy as np
import math


# First set of numerical experiments: Continuous exposure, measurements every 4 hrs, data simulated from stochastic model.
# Goal: Investigate how time length and dose selection impacts identifiability.
# (Doses are shown as a proportion of the IC50 dose for the drug.)

sim_10day_const = utils.MeasurementType(
    change_times = [0, 5 * 24],
    meas_times = [4 * k for k in range(0, round(10*24/4) + 1)],
    doses = np.array([[0, 0], 
                      [1, 0], 
                      [5, 0], 
                      [10, 0],  
                      [50, 0], 
                      [100, 0]] * 3)
)

sim_5on_5off = utils.MeasurementType(
    change_times = [0],
    meas_times = [4 * k for k in range(0, round(10*24/4) + 1)],
    doses = np.array([[0], 
                      [1], 
                      [5], 
                      [10],  
                      [50], 
                      [100]] * 3)
)


lin_param_default_no_h_nu = utils.LastYearParamSetLinear_no_h_nu(
    mu = 0.0004,
    h_mu = 0.00004,
    nu = 0.004,
    b0 = 0.04,
    d0 = 0,
    d_d0 = 0.08,
    b1 = 0.001,
    d1 = 0
)

new_param_test = utils.ParamSet_MMmu(
    mu = 0.0004,
    d_mu = 0.004,
    e_mu = 10,
    nu = 0.004,
    b0 = 0.04,
    d0 = 0,
    d_d0 = 0.08,
    b1 = 0.001,
    d1 = 0
)

n_exp = 30

# list the experments
f0_init_linear = utils.equilibf0(utils.get_fund_param_set(lin_param_default_no_h_nu, 0))
f0_init_general = utils.equilibf0(utils.get_fund_param_set(new_param_test, 0))

assert(abs(f0_init_linear - f0_init_general) < 1e-8)

experiment_inputs = [
    (n_exp, 1000, f0_init_general, True, 6, sim_10day_const, new_param_test, "general mu, const dose", 0.05, 'RMS-growth', 'equilib', False),
    (n_exp, 1000, f0_init_linear, True, 6, sim_10day_const, lin_param_default_no_h_nu, "linear mu, const dose", 0.05, 'RMS-growth', 'equilib', False),
    (n_exp, 1000, f0_init_general, True, 6, sim_5on_5off, new_param_test, "general mu, pulsed dose", 0.05, 'RMS-growth', 'equilib', False),
    (n_exp, 1000, f0_init_linear, True, 6, sim_5on_5off, lin_param_default_no_h_nu, "linear mu, pulsed dose", 0.05, 'RMS-growth', 'equilib', False),
]

#experiment_inputs = [
#    (n_exp, 1000, f0_init, True, n_hop, sim_2_5, lin_param_default_no_h_nu, "set 2 exp 5", 0.05, 'RMS-growth', 'equilib', False),
#]

# iterate through all experiments and run them
for i, tup in enumerate(experiment_inputs):
    print(f"============== starting experiment {i} ==================")
    n_exp, cell_cnt, f0, paralell, n_hops, sim_type, param_reg, msg, meas_sigma, mle_vers, f0_strat, det_simp = tup
    fitting_experiments.run_and_save_experiment(
        param_reg, sim_type, n_exp, cell_cnt, f0, paralell=paralell, 
        n_basin_hops=n_hops, message=msg, f0_strat=f0_strat, deterministic_simp=det_simp
    )

