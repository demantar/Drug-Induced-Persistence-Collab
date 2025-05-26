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
    h_nu = -0.00000,
    b0 = 0.04,
    d0 = 0.0,
    d_d0 = 0.08,
    b1 = 0.001,
    d1 = 0.0
)

experiment_inputs = [
    (20, True, 2, sim_type_1, lin_param_default, "pulsed 3 repl"), 
    (20, True, 2, sim_type_2, lin_param_default, "fixed 3 repl"),
    (20, True, 2, sim_type_1, lin_param_default, "fixed 3 repl - h_mu = 0")
]

for i, tup in enumerate(experiment_inputs):
    print(f"============== starting experiment {i} ==================")
    n_exp, paralell, n_hops, sim_type, param_reg, msg = tup
    fitting_experiments.run_and_save_experiment(
        param_reg, sim_type, n_exp, paralell=paralell, n_basin_hops=n_hops, message=msg 
    )

