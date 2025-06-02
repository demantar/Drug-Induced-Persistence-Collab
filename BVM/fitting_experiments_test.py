import fitting_experiments
import model_utils as utils
import numpy as np

sim_type_1 = utils.MeasurementTypePulsed(
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

sim_type_2 = utils.MeasurementTypePulsed(
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

sim_type_1_smaller_doses = utils.MeasurementTypePulsed(
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

sim_type_2_smaller_doses = utils.MeasurementTypePulsed(
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

sim_type_1_long = utils.MeasurementTypePulsed(
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

sim_type_2_long = utils.MeasurementTypePulsed(
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

sim_type_1_smaller_doses_long = utils.MeasurementTypePulsed(
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

sim_type_2_smaller_doses_long = utils.MeasurementTypePulsed(
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

sim_type_pulsed_long_2repl_every30 = utils.MeasurementTypePulsed(
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

sim_type_fixed_long_2repl_every30 = utils.MeasurementTypePulsed(
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

sim_type_mixed_long_2repl_every30 = utils.MeasurementTypePulsed(
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

sim_type_pulsed_long_2repl_every10 = utils.MeasurementTypePulsed(
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

sim_type_fixed_long_2repl_every10 = utils.MeasurementTypePulsed(
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

sim_type_mixed_long_2repl_every10 = utils.MeasurementTypePulsed(
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

sim_type_pulsed_long_6repl_every30 = utils.MeasurementTypePulsed(
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

sim_type_fixed_long_6repl_every30 = utils.MeasurementTypePulsed(
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

sim_type_mixed_long_6repl_every30 = utils.MeasurementTypePulsed(
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

sim_type_pulsed_long_6repl_every10 = utils.MeasurementTypePulsed(
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

sim_type_fixed_long_6repl_every10 = utils.MeasurementTypePulsed(
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

sim_type_mixed_long_6repl_every10 = utils.MeasurementTypePulsed(
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

lin_param_default_large_h_nu = utils.LastYearParamSetLinearBD(
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

lin_param_default_no_h_nu = utils.LastYearParamSetLinearBD_no_h_nu(
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

experiment_inputs_jun26 = [
    #(n_exp, True, 2, sim_type_1, lin_param_default, "pulsed 3 repl"), 
    #(n_exp, True, 2, sim_type_2, lin_param_default, "fixed 3 repl"),
    (n_exp, True, 2, sim_type_1, lin_param_default, "normal pulsed", 0.05),
    (n_exp, True, 10, sim_type_1, lin_param_default, "normal pulsed + 10 minimizations", 0.05),
    (n_exp, True, 2, sim_type_1, lin_param_default_no_h_nu, "normal pulsed without h_nu param", 0.05),
    (n_exp, True, 2, sim_type_1, lin_param_default, "normal pulsed no meas error", 0.00),
    (n_exp, True, 2, sim_type_1_smaller_doses, lin_param_default_large_h_nu, "normal pulsed with low dose, large h_nu", 0.05),
    (n_exp, True, 2, sim_type_2, lin_param_default, "normal fixed", 0.05),
    (n_exp, True, 2, sim_type_1_long, lin_param_default, "normal pulsed (longer)", 0.05),
    (n_exp, True, 10, sim_type_1_long, lin_param_default, "normal pulsed + 10 minimizations (longer)", 0.05),
    (n_exp, True, 2, sim_type_1_long, lin_param_default_no_h_nu, "normal pulsed without h_nu param (longer)", 0.05),
    (n_exp, True, 2, sim_type_1_long, lin_param_default, "normal pulsed no meas error (longer)", 0.00),
    (n_exp, True, 2, sim_type_1_smaller_doses_long, lin_param_default_large_h_nu, "normal pulsed with low dose, large h_nu (longer)", 0.05),
    (n_exp, True, 2, sim_type_2_long, lin_param_default, "normal fixed (longer)", 0.05)
]

n_exp = 30
n_hop = 4
experiment_inputs = [
    (n_exp, True, n_hop, sim_type_1_long, lin_param_default, "normal pulsed (longer) - rms-growth", 0.05, 'RMS-growth'),
    (n_exp, True, n_hop, sim_type_1_long, lin_param_default, "normal pulsed (longer) - rms-growthrate", 0.05, 'RMS-growthrate'),
    (n_exp, True, n_hop, sim_type_1_long, lin_param_default, "normal pulsed (longer) - new MLE", 0.05, 'new'),
    (n_exp, True, n_hop, sim_type_1_long, lin_param_default, "normal pulsed (longer) - rms-growth", 0.00001, 'RMS-growth'),
    (n_exp, True, n_hop, sim_type_1_long, lin_param_default, "normal pulsed (longer) - rms-growthrate", 0.00001, 'RMS-growthrate'),
    (n_exp, True, n_hop, sim_type_1_long, lin_param_default, "normal pulsed (longer) - new MLE", 0.00001, 'new')
]

experiment_inputs = [
    (n_exp, True, n_hop, sim_type_pulsed_long_2repl_every30, lin_param_default, "normal pulsed (longer) - normal-rms, 2 replicates, every 30 sec", 0.05, 'RMS-growth'),
    (n_exp, True, n_hop, sim_type_fixed_long_2repl_every30, lin_param_default, "normal fixed (longer) - normal-rms, 2 replicates, every 30 sec", 0.05, 'RMS-growth'),
    (n_exp, True, n_hop, sim_type_mixed_long_2repl_every30, lin_param_default, "normal mixed (longer) - normal-rms, 2 replicates, every 30 sec", 0.05, 'RMS-growth'),
    (n_exp, True, n_hop, sim_type_pulsed_long_2repl_every10, lin_param_default, "normal pulsed (longer) - normal-rms, 2 replicates, every 10 sec", 0.05, 'RMS-growth'),
    (n_exp, True, n_hop, sim_type_fixed_long_2repl_every10, lin_param_default, "normal fixed (longer) - normal-rms, 2 replicates, every 10 sec", 0.05, 'RMS-growth'),
    (n_exp, True, n_hop, sim_type_mixed_long_2repl_every10, lin_param_default, "normal mixed (longer) - normal-rms, 2 replicates, every 10 sec", 0.05, 'RMS-growth'),
    (n_exp, True, n_hop, sim_type_pulsed_long_6repl_every30, lin_param_default, "normal pulsed (longer) - normal-rms, 6 replicates, every 30 sec", 0.05, 'RMS-growth'),
    (n_exp, True, n_hop, sim_type_fixed_long_6repl_every30, lin_param_default, "normal fixed (longer) - normal-rms, 6 replicates, every 30 sec", 0.05, 'RMS-growth'),
    (n_exp, True, n_hop, sim_type_mixed_long_6repl_every30, lin_param_default, "normal mixed (longer) - normal-rms, 6 replicates, every 30 sec", 0.05, 'RMS-growth'),
    (n_exp, True, n_hop, sim_type_pulsed_long_6repl_every10, lin_param_default, "normal pulsed (longer) - normal-rms, 6 replicates, every 10 sec", 0.05, 'RMS-growth'),
    (n_exp, True, n_hop, sim_type_fixed_long_6repl_every10, lin_param_default, "normal fixed (longer) - normal-rms, 6 replicates, every 10 sec", 0.05, 'RMS-growth'),
    (n_exp, True, n_hop, sim_type_mixed_long_6repl_every10, lin_param_default, "normal mixed (longer) - normal-rms, 6 replicates, every 10 sec", 0.05, 'RMS-growth'),
]

for i, tup in enumerate(experiment_inputs):
    print(f"============== starting experiment {i} ==================")
    n_exp, paralell, n_hops, sim_type, param_reg, msg, meas_sigma, mle_vers = tup
    fitting_experiments.run_and_save_experiment(
        param_reg, sim_type, n_exp, paralell=paralell, n_basin_hops=n_hops, message=msg 
    )

