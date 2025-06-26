# tests for files treatment_opt_long_term.py and treatment_opt_transient.py
import model_utils as utils
from treatment_optimization import * 
from treatment_optimization_transient_phase import *

print("test output for treatment_optimization")
strat_fixed = FixedDose(param_bounds=[(0, 10)])

new_param_test = utils.ParamSet_MMmu(
    mu = 0.0004,
    d_mu = 0.004,
    e_mu = 10.0,
    nu = 0.004,
    b0 = 0.04,
    d0 = 0,
    d_d0 = 0.08,
    b1 = 0.001,
    d1 = 0
)

lin_param_default = utils.LastYearParamSetLinear(
    mu = 0.0004,
    h_mu = 0.0004,
    nu = 0.004,
    h_nu = -0.0004,
    b0 = 0.04,
    d0 = 0,
    d_d0 = 0.08,
    b1 = 0.001,
    d1 = 0
)

print(lin_param_default)

params = lin_param_default

print("testing fixed dose strategies")
strat = strat_fixed
opt_rate, opt_par, f0_eq = strat.optimize_rate(params)
print(f'opt_f: {opt_rate}, opt_x: {opt_par}, f0_eq: {f0_eq}')
doses = np.linspace(0, 10, 300)
rate_arr = [strat.get_rate([c], params)[0] for c in doses]
fig = px.line(x = doses, y = rate_arr)
fig.show()

trajectory_opt = TransientPhaseOptimizer(params, 0.98, f0_eq, opt_rate, 10, 100)

trans_obj = [trajectory_opt.transient_objective(0.5, c) for c in doses]
fig = px.line(x = doses, y = trans_obj)
fig.show()

trajectory_opt.calculate_trajectory()
fig = px.line(x = trajectory_opt.f0_points, y = trajectory_opt.dose_f0)
fig.show()

fig = px.line(x = trajectory_opt.time_points, y = trajectory_opt.dose_f0)
fig.show()
