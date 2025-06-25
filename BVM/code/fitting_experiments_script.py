# A file for queuing and saving simulated experiments
import fitting_experiments
import model_utils as utils
import numpy as np
import math


"""
First set of numerical experiments: Continuous exposure, measurements every 4 hrs, data simulated from stochastic model.
Goal: Investigate how time length and dose selection impacts identifiability.
(Doses are shown as a proportion of the IC50 dose for the drug.)

1.   T = 120 hrs (5 days), D = [0, 1, 5, 10, 50, 100].
2.   T = 120 hrs, D = [0, 0.01, 0.1, 1, 10, 100].
3.   T = 120 hrs, D = [0, 0.01, 0.05, 0.1, 0.5, 1].
4.   T = 336 hrs (14 days), D = [0, 1, 5, 10, 50, 100].
5.   T = 336 hrs, D = [0, 0.01, 0.1, 1, 10, 100].
6.   T = 336 hrs, D =  [0, 0.01, 0.1, 1, 10, 100].

Second set of numerical experiments: Continuous vs pulsed exposure, measurements every 4 hrs, data simulated from stochastic model.
Goal: Compare identifiability under continuous and intermittent data. Time length and dose selection also considered.

1.   T = 240 hrs (10 days), continuous exposure, D = [0, 1, 5, 10, 50, 100].
2.   T = 240 hrs, continuous exposure, D = [0, 0.01, 0.1, 1, 10, 100].
3.   T = 240 hrs, intermittent (5+5), D =  [0, 1, 5, 10, 50, 100].
4.   T = 240 hrs, intermittent (5+5), D = [0, 0.01, 0.1, 1, 10, 100].
5.   T = 480 hrs (20 days), continuous exposure, D =  [0, 1, 5, 10, 50, 100].
6.   T = 480 hrs, continuous exposure, D = [0, 0.01, 0.1, 1, 10, 100].
7.   T = 480 hrs, intermittent (5+5+5+5), D = [0, 1, 5, 10, 50, 100].
8.   T = 480 hrs, intermittent (5+5+5+5), D = [0, 0.01, 0.1, 1, 10, 100].

Third set of numerical experiments: Structural identifiability analysis with deterministic data.  Continuous exposure.
Goal: Check if parameters can be extracted given perfect data.

1.   T = 240 hrs, D = [0, 1, 5, 10, 50, 100], no measurement error.
2.   T = 240 hrs, D = [0, 1, 2, 5, 10, 20, 50, 75, 100], no measurement error.
3.   T = 240 hrs, D = [0, 1, 5, 10, 50, 100], with measurement error.
4.   T = 240 hrs, D = [0, 1, 2, 5, 10, 20, 50, 75, 100], with measurement error.
5.   T = 480 hrs, D = [0, 1, 5, 10, 50, 100], no measurement error.
6.   T = 480 hrs, D = [0, 1, 2, 5, 10, 20, 50, 75, 100], no measurement error.
7.   T = 480 hrs, D = [0, 1, 5, 10, 50, 100], with measurement error.
8.   T = 480 hrs, D = [0, 1, 2, 5, 10, 20, 50, 75, 100], with measurement error.
"""


# First set of numerical experiments: Continuous exposure, measurements every 4 hrs, data simulated from stochastic model.
# Goal: Investigate how time length and dose selection impacts identifiability.
# (Doses are shown as a proportion of the IC50 dose for the drug.)

# 1.   T = 120 hrs (5 days), D = [0, 1, 5, 10, 50, 100].
sim_1_1 = utils.MeasurementType(
    change_times = [0],
    meas_times = [4 * k for k in range(0, round(120/4) + 1)],
    doses = np.array([[0], 
                      [1], 
                      [5], 
                      [10],  
                      [50], 
                      [100]] * 3)
)

# 2.   T = 120 hrs, D = [0, 0.01, 0.1, 1, 10, 100].
sim_1_2 = utils.MeasurementType(
    change_times = [0],
    meas_times = [4 * k for k in range(0, round(120/4) + 1)],
    doses = np.array([[0], 
                      [0.01],
                      [0.1], 
                      [1],
                      [50],
                      [100]] * 3)
)

# 3.   T = 120 hrs, D = [0, 0.01, 0.05, 0.1, 0.5, 1].
sim_1_3 = utils.MeasurementType(
    change_times = [0],
    meas_times = [4 * k for k in range(0, round(120/4) + 1)],
    doses = np.array([[0], 
                      [0.01],
                      [0.05], 
                      [0.1],
                      [0.5],
                      [1]] * 3)
)

# 4.   T = 336 hrs (14 days), D = [0, 1, 5, 10, 50, 100].
sim_1_4 = utils.MeasurementType(
    change_times = [0],
    meas_times = [4 * k for k in range(0, round(336/4) + 1)],
    doses = np.array([[0], 
                      [1], 
                      [5], 
                      [10],  
                      [50], 
                      [100]] * 3)
)

sim_1_4_mod = utils.MeasurementType(
    change_times = [0],
    meas_times = [4 * k for k in range(0, round(336/4) + 1)],
    doses = np.array([[0.1], 
                      [5], 
                      [10],  
                      [50], 
                      [100]] * 3)
)

# 5.   T = 336 hrs, D = [0, 0.01, 0.1, 1, 10, 100].
sim_1_5 = utils.MeasurementType(
    change_times = [0],
    meas_times = [4 * k for k in range(0, round(336/4) + 1)],
    doses = np.array([[0], 
                      [0.01],
                      [0.1], 
                      [1],
                      [50],
                      [100]] * 3)
)

# 6.   T = 336 hrs, D = [0, 0.01, 0.05, 0.1, 0.5, 1].
sim_1_6 = utils.MeasurementType(
    change_times = [0],
    meas_times = [4 * k for k in range(0, round(336/4) + 1)],
    doses = np.array([[0], 
                      [0.01],
                      [0.05], 
                      [0.1],
                      [0.5],
                      [1]] * 3)
)

# Second set of numerical experiments: Continuous vs pulsed exposure, measurements every 4 hrs, data simulated from stochastic model.
# Goal: Compare identifiability under continuous and intermittent data. Time length and dose selection also considered.

# 1.   T = 240 hrs (10 days), continuous exposure, D = [0, 1, 5, 10, 50, 100].
sim_2_1 = utils.MeasurementType(
    change_times = [0],
    meas_times = [4 * k for k in range(0, round(240/4) + 1)],
    doses = np.array([[0], 
                      [1], 
                      [5], 
                      [10],  
                      [50], 
                      [100]] * 3)
)

# 2.   T = 240 hrs, continuous exposure, D = [0, 0.01, 0.1, 1, 10, 100].
sim_2_2 = utils.MeasurementType(
    change_times = [0],
    meas_times = [4 * k for k in range(0, round(240/4) + 1)],
    doses = np.array([[0], 
                      [0.01], 
                      [0.1], 
                      [1],  
                      [10], 
                      [100]] * 3)
)

# 3.   T = 240 hrs, intermittent (5+5), D =  [0, 1, 5, 10, 50, 100].
sim_2_3 = utils.MeasurementType(
    change_times = [0, 120],
    meas_times = [4 * k for k in range(0, round(240/4) + 1)],
    doses = np.array([[0, 0], 
                      [1, 0], 
                      [5, 0], 
                      [10, 0],  
                      [50, 0], 
                      [100, 0]] * 3)
)

# 4.   T = 240 hrs, intermittent (5+5), D = [0, 0.01, 0.1, 1, 10, 100].
sim_2_4 = utils.MeasurementType(
    change_times = [0, 120],
    meas_times = [4 * k for k in range(0, round(240/4) + 1)],
    doses = np.array([[0, 0], 
                      [0.01, 0], 
                      [0.1, 0], 
                      [1, 0],  
                      [10, 0], 
                      [100, 0]] * 3)
)

# 5.   T = 480 hrs (20 days), continuous exposure, D =  [0, 1, 5, 10, 50, 100].
sim_2_5 = utils.MeasurementType(
    change_times = [0],
    meas_times = [4 * k for k in range(0, round(480/4) + 1)],
    doses = np.array([[0], 
                      [1], 
                      [5], 
                      [10],  
                      [50], 
                      [100]] * 3)
)

# 6.   T = 480 hrs, continuous exposure, D = [0, 0.01, 0.1, 1, 10, 100].
sim_2_6 = utils.MeasurementType(
    change_times = [0],
    meas_times = [4 * k for k in range(0, round(480/4) + 1)],
    doses = np.array([[0], 
                      [0.01], 
                      [0.1], 
                      [1],  
                      [10], 
                      [100]] * 3)
)

# 7.   T = 480 hrs, intermittent (5+5+5+5), D = [0, 1, 5, 10, 50, 100].
sim_2_7 = utils.MeasurementType(
    change_times = [0, 120, 240, 360],
    meas_times = [4 * k for k in range(0, round(480/4) + 1)],
    doses = np.array([[0, 0, 0, 0], 
                      [1, 0, 1, 0], 
                      [5, 0, 5, 0], 
                      [10, 0, 10, 0],  
                      [50, 0, 50, 0], 
                      [100, 0, 100, 0]] * 3)
)

# 8.   T = 480 hrs, intermittent (5+5+5+5), D = [0, 0.01, 0.1, 1, 10, 100].
sim_2_8 = utils.MeasurementType(
    change_times = [0, 120, 240, 360],
    meas_times = [4 * k for k in range(0, round(480/4) + 1)],
    doses = np.array([[0, 0, 0, 0], 
                      [0.01, 0, 0.01, 0], 
                      [0.1, 0, 0.1, 0], 
                      [1, 0, 1, 0],  
                      [10, 0, 10, 0], 
                      [100, 0, 100, 0]] * 3)
)

# Third set of numerical experiments: Structural identifiability analysis with deterministic data.  Continuous exposure.
# Goal: Check if parameters can be extracted given perfect data.

# 1.   T = 240 hrs, D = [0, 1, 5, 10, 50, 100], no measurement error.
# 3.   T = 240 hrs, D = [0, 1, 5, 10, 50, 100], with measurement error.
sim_3_13 = utils.MeasurementType(
    change_times = [0],
    meas_times = [4 * k for k in range(0, round(240/4) + 1)],
    doses = np.array([[0], 
                      [1], 
                      [5], 
                      [10],  
                      [50], 
                      [100]] * 3)
)

# 2.   T = 240 hrs, D = [0, 1, 2, 5, 10, 20, 50, 75, 100], no measurement error.
# 4.   T = 240 hrs, D = [0, 1, 2, 5, 10, 20, 50, 75, 100], with measurement error.
sim_3_24 = utils.MeasurementType(
    change_times = [0],
    meas_times = [4 * k for k in range(0, round(240/4) + 1)],
    doses = np.array([[0], 
                      [1], 
                      [2], 
                      [5], 
                      [10],  
                      [20],  
                      [50], 
                      [75], 
                      [100]] * 3)
)

# 5.   T = 480 hrs, D = [0, 1, 5, 10, 50, 100], no measurement error.
# 7.   T = 480 hrs, D = [0, 1, 5, 10, 50, 100], with measurement error.
sim_3_57 = utils.MeasurementType(
    change_times = [0],
    meas_times = [4 * k for k in range(0, round(480/4) + 1)],
    doses = np.array([[0], 
                      [1], 
                      [5], 
                      [10],  
                      [50], 
                      [100]] * 3)
)

# 6.   T = 480 hrs, D = [0, 1, 2, 5, 10, 20, 50, 75, 100], no measurement error.
# 8.   T = 480 hrs, D = [0, 1, 2, 5, 10, 20, 50, 75, 100], with measurement error.
sim_3_68 = utils.MeasurementType(
    change_times = [0],
    meas_times = [4 * k for k in range(0, round(480/4) + 1)],
    doses = np.array([[0], 
                      [1], 
                      [2], 
                      [5], 
                      [10],  
                      [20],  
                      [50], 
                      [75], 
                      [100]] * 3)
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

lin_param_default_logged = utils.LastYearParamSetLinear_logged(
    mu = math.log(0.0004),
    h_mu = math.log(0.00004),
    nu = math.log(0.004),
    h_nu = math.log(0.00004),
    b0 = math.log(0.04),
    d0 = math.log(1e-8),
    d_d0 = math.log(0.08),
    b1 = math.log(0.001),
    d1 = math.log(1e-8)
)

lin_param_default_no_h_nu_logged = utils.LastYearParamSetLinear_no_h_nu_logged(
    mu = math.log(0.0004),
    h_mu = math.log(0.00004),
    nu = math.log(0.004),
    b0 = math.log(0.04),
    d0 = math.log(1e-8),
    d_d0 = math.log(0.08),
    b1 = math.log(0.001),
    d1 = math.log(1e-8)
)

n_exp = 16

# list the experments
f0_init = utils.equilibf0(utils.get_fund_param_set(lin_param_default_no_h_nu, 0))

experiment_inputs = [
    (n_exp, 1000, f0_init, True, 5, sim_1_1, lin_param_default_no_h_nu, "set 1 exp 1", 0.05, 'RMS-growth', 'equilib', False),
    (n_exp, 1000, f0_init, True, 5, sim_1_2, lin_param_default_no_h_nu, "set 1 exp 2", 0.05, 'RMS-growth', 'equilib', False),
    (n_exp, 1000, f0_init, True, 5, sim_1_3, lin_param_default_no_h_nu, "set 1 exp 3", 0.05, 'RMS-growth', 'equilib', False),
    (n_exp, 1000, f0_init, True, 5, sim_1_4, lin_param_default_no_h_nu, "set 1 exp 4", 0.05, 'RMS-growth', 'equilib', False),
    (n_exp, 1000, f0_init, True, 5, sim_1_5, lin_param_default_no_h_nu, "set 1 exp 5", 0.05, 'RMS-growth', 'equilib', False),
    (n_exp, 1000, f0_init, True, 5, sim_1_6, lin_param_default_no_h_nu, "set 1 exp 6", 0.05, 'RMS-growth', 'equilib', False),

    (n_exp, 1000, f0_init, True, 3, sim_2_1, lin_param_default_no_h_nu, "set 2 exp 1", 0.05, 'RMS-growth', 'equilib', False),
    (n_exp, 1000, f0_init, True, 3, sim_2_2, lin_param_default_no_h_nu, "set 2 exp 2", 0.05, 'RMS-growth', 'equilib', False),
    (n_exp, 1000, f0_init, True, 3, sim_2_3, lin_param_default_no_h_nu, "set 2 exp 3", 0.05, 'RMS-growth', 'equilib', False),
    (n_exp, 1000, f0_init, True, 3, sim_2_4, lin_param_default_no_h_nu, "set 2 exp 4", 0.05, 'RMS-growth', 'equilib', False),
    (n_exp, 1000, f0_init, True, 3, sim_2_5, lin_param_default_no_h_nu, "set 2 exp 5", 0.05, 'RMS-growth', 'equilib', False),
    (n_exp, 1000, f0_init, True, 3, sim_2_6, lin_param_default_no_h_nu, "set 2 exp 6", 0.05, 'RMS-growth', 'equilib', False),
    (n_exp, 1000, f0_init, True, 3, sim_2_7, lin_param_default_no_h_nu, "set 2 exp 7", 0.05, 'RMS-growth', 'equilib', False),
    (n_exp, 1000, f0_init, True, 3, sim_2_8, lin_param_default_no_h_nu, "set 2 exp 8", 0.05, 'RMS-growth', 'equilib', False),

    (n_exp, 1000, f0_init, True, 2, sim_3_13, lin_param_default_no_h_nu, "set 3 exp 1", 0.0, 'RMS-growth', 'equilib', True),
    (n_exp, 1000, f0_init, True, 2, sim_3_24, lin_param_default_no_h_nu, "set 3 exp 2", 0.05, 'RMS-growth', 'equilib', True),
    (n_exp, 1000, f0_init, True, 2, sim_3_13, lin_param_default_no_h_nu, "set 3 exp 3", 0.0, 'RMS-growth', 'equilib', True),
    (n_exp, 1000, f0_init, True, 2, sim_3_24, lin_param_default_no_h_nu, "set 3 exp 4", 0.05, 'RMS-growth', 'equilib', True),
    (n_exp, 1000, f0_init, True, 2, sim_3_57, lin_param_default_no_h_nu, "set 3 exp 5", 0.0, 'RMS-growth', 'equilib', True),
    (n_exp, 1000, f0_init, True, 2, sim_3_68, lin_param_default_no_h_nu, "set 3 exp 6", 0.05, 'RMS-growth', 'equilib', True),
    (n_exp, 1000, f0_init, True, 2, sim_3_57, lin_param_default_no_h_nu, "set 3 exp 7", 0.0, 'RMS-growth', 'equilib', True),
    (n_exp, 1000, f0_init, True, 2, sim_3_68, lin_param_default_no_h_nu, "set 3 exp 8", 0.05, 'RMS-growth', 'equilib', True),
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

