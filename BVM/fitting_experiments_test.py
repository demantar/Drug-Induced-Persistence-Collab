# A file for queuing and saving simulated experiments
import fitting_experiments
import model_utils as utils
import numpy as np

# set up measurement types for simulations
sim_type_1 = utils.MeasurementType(
    change_times = [0, 50],
    meas_times = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
    doses = np.array([[1, 0], [1, 0], [1, 0],
                      [5, 0], [5, 0], [5, 0],
                      [10, 0], [10, 0], [10, 0],
                      [20, 0], [20, 0], [20, 0], 
                      [50, 0], [50, 0], [50, 0],
                      [75, 0], [75, 0], [75, 0],
                      [100, 0], [100, 0], [100, 0]])
)

sim_type_2 = utils.MeasurementType(
    change_times = [0],
    meas_times = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
    doses = np.array([[1], [1], [1],
                      [5], [5], [5],
                      [10], [10], [10],
                      [20], [20], [20], 
                      [50], [50], [50],
                      [75], [75], [75],
                      [100], [100], [100]])
)

sim_type_1_smaller_doses = utils.MeasurementType(
    change_times = [0, 50],
    meas_times = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
    doses = np.array([[1, 0], [1, 0], [1, 0],
                      [5, 0], [5, 0], [5, 0],
                      [10, 0], [10, 0], [10, 0],
                      [20, 0], [20, 0], [20, 0], 
                      [50, 0], [50, 0], [50, 0],
                      [75, 0], [75, 0], [75, 0],
                      [100, 0], [100, 0], [100, 0]]) / 10
)

sim_type_2_smaller_doses = utils.MeasurementType(
    change_times = [0],
    meas_times = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
    doses = np.array([[1], [1], [1],
                      [5], [5], [5],
                      [10], [10], [10],
                      [20], [20], [20], 
                      [50], [50], [50],
                      [75], [75], [75],
                      [100], [100], [100]]) / 10
)

sim_type_1_long = utils.MeasurementType(
    change_times = [0, 150],
    meas_times = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300],
    doses = np.array([[1, 0], [1, 0], [1, 0],
                      [5, 0], [5, 0], [5, 0],
                      [10, 0], [10, 0], [10, 0],
                      [20, 0], [20, 0], [20, 0], 
                      [50, 0], [50, 0], [50, 0],
                      [75, 0], [75, 0], [75, 0],
                      [100, 0], [100, 0], [100, 0]])
)

sim_type_2_long = utils.MeasurementType(
    change_times = [0],
    meas_times = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300],
    doses = np.array([[1], [1], [1],
                      [5], [5], [5],
                      [10], [10], [10],
                      [20], [20], [20], 
                      [50], [50], [50],
                      [75], [75], [75],
                      [100], [100], [100]])
)

sim_type_1_smaller_doses_long = utils.MeasurementType(
    change_times = [0, 150],
    meas_times = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300],
    doses = np.array([[1, 0], [1, 0], [1, 0],
                      [5, 0], [5, 0], [5, 0],
                      [10, 0], [10, 0], [10, 0],
                      [20, 0], [20, 0], [20, 0], 
                      [50, 0], [50, 0], [50, 0],
                      [75, 0], [75, 0], [75, 0],
                      [100, 0], [100, 0], [100, 0]]) / 10
)

sim_type_2_smaller_doses_long = utils.MeasurementType(
    change_times = [0],
    meas_times = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300],
    doses = np.array([[1], [1], [1],
                      [5], [5], [5],
                      [10], [10], [10],
                      [20], [20], [20], 
                      [50], [50], [50],
                      [75], [75], [75],
                      [100], [100], [100]]) / 10
)

sim_type_pulsed_long_2repl_every30 = utils.MeasurementType(
    change_times = [0, 150],
    meas_times = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300],
    doses = np.array([[1, 0], [1, 0],
                      [5, 0], [5, 0],
                      [10, 0], [10, 0],
                      [20, 0], [20, 0], 
                      [50, 0], [50, 0],
                      [75, 0], [75, 0],
                      [100, 0], [100, 0]])
)

sim_type_fixed_long_2repl_every30 = utils.MeasurementType(
    change_times = [0],
    meas_times = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300],
    doses = np.array([[1], [1],
                      [5], [5],
                      [10], [10],
                      [20], [20], 
                      [50], [50],
                      [75], [75],
                      [100], [100]])
)

sim_type_mixed_long_2repl_every30 = utils.MeasurementType(
    change_times = [0, 150],
    meas_times = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300],
    doses = np.array([[1, 1], [1, 0],
                      [5, 5], [5, 0],
                      [10, 10], [10, 0],
                      [20, 20], [20, 0], 
                      [50, 50], [50, 0],
                      [75, 75], [75, 0],
                      [100, 100], [100, 0]])
)

sim_type_pulsed_long_2repl_every10 = utils.MeasurementType(
    change_times = [0, 150],
    meas_times = [10 * k for k in range(31)],
    doses = np.array([[1, 0], [1, 0],
                      [5, 0], [5, 0],
                      [10, 0], [10, 0],
                      [20, 0], [20, 0], 
                      [50, 0], [50, 0],
                      [75, 0], [75, 0],
                      [100, 0], [100, 0]])
)

sim_type_fixed_long_2repl_every10 = utils.MeasurementType(
    change_times = [0],
    meas_times = [10 * k for k in range(31)],
    doses = np.array([[1], [1],
                      [5], [5],
                      [10], [10],
                      [20], [20], 
                      [50], [50],
                      [75], [75],
                      [100], [100]])
)

sim_type_mixed_long_2repl_every10 = utils.MeasurementType(
    change_times = [0, 150],
    meas_times = [10 * k for k in range(31)],
    doses = np.array([[1, 1], [1, 0],
                      [5, 5], [5, 0],
                      [10, 10], [10, 0],
                      [20, 20], [20, 0], 
                      [50, 50], [50, 0],
                      [75, 75], [75, 0],
                      [100, 100], [100, 0]])
)

sim_type_pulsed_long_6repl_every30 = utils.MeasurementType(
    change_times = [0, 150],
    meas_times = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300],
    doses = np.array([[1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0],
                      [5, 0], [5, 0], [5, 0], [5, 0], [5, 0], [5, 0],
                      [10, 0], [10, 0], [10, 0], [10, 0], [10, 0], [10, 0],
                      [20, 0], [20, 0], [20, 0], [20, 0], [20, 0], [20, 0],
                      [50, 0], [50, 0], [50, 0], [50, 0], [50, 0], [50, 0],
                      [75, 0], [75, 0], [75, 0], [75, 0], [75, 0], [75, 0],
                      [100, 0], [100, 0], [100, 0], [100, 0], [100, 0], [100, 0]])
)

sim_type_fixed_long_6repl_every30 = utils.MeasurementType(
    change_times = [0],
    meas_times = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300],
    doses = np.array([[1], [1], [1], [1], [1], [1],
                      [5], [5], [5], [5], [5], [5],
                      [10], [10], [10], [10], [10], [10],
                      [20], [20], [20], [20], [20], [20],
                      [50], [50], [50], [50], [50], [50],
                      [75], [75], [75], [75], [75], [75],
                      [100], [100], [100], [100], [100], [100]])
)

sim_type_mixed_long_6repl_every30 = utils.MeasurementType(
    change_times = [0, 150],
    meas_times = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300],
    doses = np.array([[1, 1], [1, 0], [1, 1], [1, 0], [1, 1], [1, 0],
                      [5, 5], [5, 0], [5, 5], [5, 0], [5, 5], [5, 0],
                      [10, 10], [10, 0], [10, 10], [10, 0], [10, 10], [10, 0],
                      [20, 20], [20, 0], [20, 20], [20, 0], [20, 20], [20, 0],
                      [50, 50], [50, 0], [50, 50], [50, 0], [50, 50], [50, 0],
                      [75, 75], [75, 0], [75, 75], [75, 0], [75, 75], [75, 0],
                      [100, 100], [100, 0], [100, 100], [100, 0], [100, 100], [100, 0]])
)

sim_type_pulsed_long_6repl_every10 = utils.MeasurementType(
    change_times = [0, 150],
    meas_times = [10 * k for k in range(31)],
    doses = np.array([[1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0],
                      [5, 0], [5, 0], [5, 0], [5, 0], [5, 0], [5, 0],
                      [10, 0], [10, 0], [10, 0], [10, 0], [10, 0], [10, 0],
                      [20, 0], [20, 0], [20, 0], [20, 0], [20, 0], [20, 0], 
                      [50, 0], [50, 0], [50, 0], [50, 0], [50, 0], [50, 0],
                      [75, 0], [75, 0], [75, 0], [75, 0], [75, 0], [75, 0],
                      [100, 0], [100, 0], [100, 0], [100, 0], [100, 0], [100, 0]])
)

sim_type_fixed_long_6repl_every10 = utils.MeasurementType(
    change_times = [0],
    meas_times = [10 * k for k in range(31)],
    doses = np.array([[1], [1], [1], [1], [1], [1],
                      [5], [5], [5], [5], [5], [5],
                      [10], [10], [10], [10], [10], [10],
                      [20], [20], [20], [20], [20], [20],
                      [50], [50], [50], [50], [50], [50],
                      [75], [75], [75], [75], [75], [75],
                      [100], [100], [100], [100], [100], [100]])
)

sim_type_mixed_long_6repl_every10 = utils.MeasurementType(
    change_times = [0, 150],
    meas_times = [10 * k for k in range(31)],
    doses = np.array([[1, 1], [1, 0], [1, 1], [1, 0], [1, 1], [1, 0],
                      [5, 5], [5, 0], [5, 5], [5, 0], [5, 5], [5, 0],
                      [10, 10], [10, 0], [10, 10], [10, 0], [10, 10], [10, 0],
                      [20, 20], [20, 0], [20, 20], [20, 0], [20, 20], [20, 0],
                      [50, 50], [50, 0], [50, 50], [50, 0], [50, 50], [50, 0],
                      [75, 75], [75, 0], [75, 75], [75, 0], [75, 75], [75, 0],
                      [100, 100], [100, 0], [100, 100], [100, 0], [100, 100], [100, 0]])
)

# make instances of parameter regimes to test against
lin_param_default = utils.LastYearParamSetLinear(
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

lin_param_default = utils.LastYearParamSetLinear(
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

lin_param_default_large_h_nu = utils.LastYearParamSetLinear(
    mu = 0.0004,
    h_mu = 0.00004,
    nu = 0.004,
    h_nu = -0.0004,
    b0 = 0.04,
    d0 = 0.0,
    d_d0 = 0.08,
    b1 = 0.001,
    d1 = 0.0
)

lin_param_default_no_h_nu = utils.LastYearParamSetLinear_no_h_nu(
    mu = 0.0004,
    h_mu = 0.00004,
    nu = 0.004,
    b0 = 0.04,
    d0 = 0.0,
    d_d0 = 0.08,
    b1 = 0.001,
    d1 = 0.0
)

n_exp = 20
n_hop = 5

# list the experments
f0_init = utils.equilibf0(utils.get_fund_param_set(lin_param_default, 0))
print(f0_init)
experiment_inputs = [
    (n_exp, 1000, f0_init, True, n_hop, sim_type_pulsed_long_2repl_every30, lin_param_default, "f0_init test: true = equilib, strat = true", 0.05, 'RMS-growth', 'true'),
    (n_exp, 1000, f0_init, True, n_hop, sim_type_pulsed_long_2repl_every30, lin_param_default, "f0_init test: true = equilib, strat = equilib", 0.05, 'RMS-growth', 'equilib'),
    (n_exp, 1000, f0_init, True, n_hop, sim_type_pulsed_long_2repl_every30, lin_param_default, "f0_init test: true = equilib, strat = =1", 0.05, 'RMS-growth', '=1'),
    (n_exp, 1000, f0_init, True, n_hop, sim_type_pulsed_long_2repl_every30, lin_param_default, "f0_init test: true = equilib, strat = fit", 0.05, 'RMS-growth', 'fit'),
    (n_exp, 1000, 0.999, True, n_hop, sim_type_pulsed_long_2repl_every30, lin_param_default, "f0_init test: true = 1, strat = true", 0.05, 'RMS-growth', 'true'),
    (n_exp, 1000, 0.999, True, n_hop, sim_type_pulsed_long_2repl_every30, lin_param_default, "f0_init test: true = 1, strat = equilib", 0.05, 'RMS-growth', 'equilib'),
    (n_exp, 1000, 0.999, True, n_hop, sim_type_pulsed_long_2repl_every30, lin_param_default, "f0_init test: true = 1, strat = =1", 0.05, 'RMS-growth', '=1'),
    (n_exp, 1000, 0.999, True, n_hop, sim_type_pulsed_long_2repl_every30, lin_param_default, "f0_init test: true = 1, strat = fit", 0.05, 'RMS-growth', 'fit'),
]


# iterate through all experiments and run them
for i, tup in enumerate(experiment_inputs):
    print(f"============== starting experiment {i} ==================")
    n_exp, cell_cnt, f0, paralell, n_hops, sim_type, param_reg, msg, meas_sigma, mle_vers, f0_strat = tup
    fitting_experiments.run_and_save_experiment(
        param_reg, sim_type, n_exp, cell_cnt, f0, paralell=paralell, n_basin_hops=n_hops, message=msg, f0_strat=f0_strat
    )

