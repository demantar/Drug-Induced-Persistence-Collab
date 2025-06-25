import casadi as ca
import math
import plotly.express as px
import numpy as np
import model_utils as utils
import plotly.graph_objects as go
from simulate import simulate

def optimize(Xs, cs, params, opti):
    times = sim.type.meas_times
    N = len(times) - 1

    mu, h_mu, nu, b0, d0, d_d0, b1, d1 = [ca.exp(param) for param in params]

    #---------------
    # Build reusable integrator accepting dt as a parameter
    x_vec = ca.MX.sym('x', 4, 1)
    cp_sym = ca.MX.sym('cp', 1)
    param_sym = ca.MX.sym('theta', 8)
    dt_sym = ca.MX.sym('dt')

    mu_, h_mu_, nu_, b0_, d0_, d_d0_, b1_, d1_ = [param_sym[i] for i in range(8)]

    lambda0 = b0_ - (d0_ + d_d0_ * cp_sym / (cp_sym + 1))
    lambda1 = b1_ - d1_
    mu_c    = mu_ + cp_sym * h_mu_
    nu_c    = nu_

    A = ca.vertcat(
        ca.horzcat(lambda0 - mu_c, mu_c),
        ca.horzcat(nu_c, lambda1 - nu_c)
    )

    x_mat = ca.reshape(x_vec, 2, 2)

    ode = {'x': x_vec, 'p': ca.vertcat(cp_sym, param_sym, dt_sym), 'ode': dt_sym * ca.reshape(A @ x_mat, 4, 1)}

    integrator = ca.integrator('F', 'cvodes', ode, {'tf': 1.0})

    #---------------
    # Objective
    obj_dynamics = 0 # dynamics error objective
    obj_meas = 0 # measurement error objective

    theta_vec = ca.vertcat(mu, h_mu, nu, b0, d0, d_d0, b1, d1) 

    dets = []
    deltas = []
    covs = []
    obj_meas_deltas = []
    obj_dynamics_deltas = []
    for i_ds, dose_sched in enumerate(sim.type.doses):
        dets_i = []
        deltas_i = []
        covs_i = []
        obj_meas_deltas_i = []
        obj_dynamics_deltas_i = []

        obj_meas_delta = ca.log(2 * math.pi * 0.05) / 2 + (ca.log(ca.sum1(Xs[i_ds][:, 0])) - ca.log(sim.data[i_ds, 0])) ** 2 / 0.05 / 2
        obj_meas_deltas_i.append(obj_meas_delta)
        obj_meas += obj_meas_delta

        for i in range(N):
            dt_i = times[i+1] - times[i]
            #res = integrator(x0=Xs[i_ds][:, i], p=ca.vertcat(cs[i_ds][i], theta_vec, dt_i))
            res = integrator(x0=ca.reshape(ca.MX.eye(2), 4, 1), p=ca.vertcat(cs[i_ds][i], theta_vec, dt_i))
            exp_a = ca.reshape(res['xf'], 2, 2)
            X_prop = (Xs[i_ds][:,i].T @ exp_a).T

            obj_meas_delta = ca.log(2 * math.pi * 0.05) / 2 + (ca.log(ca.sum1(Xs[i_ds][:, i+1])) - ca.log(sim.data[i_ds, i+1]))**2 / 0.05 / 2  # measurement fitting
            obj_meas_deltas_i.append(obj_meas_delta)
            obj_meas += obj_meas_delta

            X0_0, X1_0 = Xs[i_ds][0, i], Xs[i_ds][1, i]
            d0_c = d0 + d_d0 * dose_sched[i] / (dose_sched[i] + 1)
            a11_0 = (X0_0 * (mu + b0 + d0_c) + X1_0 * nu) 
            a12_21_0 = (-X0_0 * mu - X1_0 * nu) 
            a22_0 = (X1_0 * (nu + b1 + d1) + X0_0 * mu) 
            Q_0 = ca.vertcat(
                ca.horzcat(a11_0, a12_21_0),
                ca.horzcat(a12_21_0, a22_0),
            )  

            X0_1, X1_1 = X_prop[0], X_prop[1]
            d0_c = d0 + d_d0 * dose_sched[i] / (dose_sched[i] + 1)
            a11_1 = (X0_1 * (mu + b0 + d0_c) + X1_1 * nu) 
            a12_21_1 = (-X0_1 * mu - X1_1 * nu) 
            a22_1 = (X1_1 * (nu + b1 + d1) + X0_1 * mu) 
            Q_1 = ca.vertcat(
                ca.horzcat(a11_1, a12_21_1),
                ca.horzcat(a12_21_1, a22_1),
            )  

            cov = dt_i / 2 * (exp_a.T @ Q_0 @ exp_a + Q_1)
            det = cov[0, 0] * cov[1, 1] - cov[1, 0] * cov[0, 1]
            cov_inv = ca.vertcat(
                ca.horzcat(cov[1, 1], -cov[0, 1]),
                ca.horzcat(-cov[1, 0], cov[0, 0])
            ) / det

            dets_i.append(det)
            delta = X_prop - Xs[i_ds][:, i+1]
            deltas_i.append(delta)
            covs_i.append(cov)

            obj_dynamics_delta = ca.log(4 * math.pi ** 2 * det) / 2 + delta.T @ cov_inv @ delta / 2  # shooting mismatch
            obj_dynamics += obj_dynamics_delta
            obj_dynamics_deltas_i.append(obj_dynamics_delta)
            #obj_dynamics += delta.T @ cov_inv @ delta / 2  # shooting mismatch
        dets.append(dets_i)
        deltas.append(deltas_i)
        covs.append(covs_i)
        obj_meas_deltas.append(obj_meas_deltas_i)
        obj_dynamics_deltas.append(obj_dynamics_deltas_i)

    # combined objective
    obj = obj_dynamics + obj_meas 

    # Set objective
    opti.minimize(obj)

    # Solver options
    p_opts = {"expand": True}
    s_opts = {"max_iter": 500, "tol": 1e-10, "print_level": 5}
    opti.solver('ipopt', p_opts, s_opts)

    try:
        sol = opti.solve()
        X_opts = [sol.value(Xs[i]) for i in range(len(sim.type.doses))]
        P_opt = (sol.value(mu), sol.value(h_mu), sol.value(nu), sol.value(b0), sol.value(d0), sol.value(d_d0), sol.value(b1), sol.value(d1))
        obj_dynamics_opt, obj_meas_opt = (sol.value(obj_dynamics), sol.value(obj_meas))
        dets = [[sol.value(dets[i][j]) for j in range(N)] for i in range(len(sim.type.doses))] 
        print("dets: ")
        print(dets)
        deltas = [[sol.value(deltas[i][j]) for j in range(N)] for i in range(len(sim.type.doses))]
        print("deltas: ")
        print(deltas)
        covs = [[sol.value(covs[i][j]) for j in range(N)] for i in range(len(sim.type.doses))]
        print("covs: ")
        print(covs)
        obj_meas_deltas = [[sol.value(obj_meas_deltas[i][j]) for j in range(N)] for i in range(len(sim.type.doses))]
        print("obj_meas_deltas: ")
        print(obj_meas_deltas)
        obj_dynamics_deltas = [[sol.value(obj_dynamics_deltas[i][j]) for j in range(N)] for i in range(len(sim.type.doses))]
        print("obj_dynamics_deltas: ")
        print(obj_dynamics_deltas)
    except RuntimeError as e:
        print("WARNING: Solver failed, returning current iterate.")
        print(e)
        X_opts = [opti.debug.value(Xs[i]) for i in range(len(sim.type.doses))]
        P_opt = (opti.debug.value(mu), opti.debug.value(h_mu), opti.debug.value(nu), opti.debug.value(b0), opti.debug.value(d0), opti.debug.value(d_d0), opti.debug.value(b1), opti.debug.value(d1))
        obj_dynamics_opt, obj_meas_opt = (opti.debug.value(obj_dynamics), opti.debug.value(obj_meas))


    print(f'dynamics objective: {obj_dynamics_opt}, meas objective: {obj_meas_opt}')

    return X_opts, P_opt

# Multiple shooting fitting states to a fixed control sequence using an ODE integrator
# Controls c_data are provided externally (length N)
def multiple_shooting_fixed_c(sim, perfect_initial=False, true_params=None):
    times = sim.type.meas_times
    N = len(times) - 1
    print(N)

    # Create Opti instance
    opti = ca.Opti()

    # Parameter: control vector
    cs = []
    Xs = []
    params = []

    # Decision variables: state nodes (2 x (N+1))
    for dose_sched, meas in zip(sim.type.doses, sim.data):
        c = opti.parameter(N)
        opti.set_value(c, dose_sched)

        X = opti.variable(2, N+1)
        opti.set_initial(X[0, :], 0.5 * meas)
        opti.set_initial(X[1, :], 0.5 * meas)
        # Initial condition
        # opti.subject_to(X[:, 0] == ca.DM(X0))

        # Constrain states to be nonnegative
        opti.subject_to(opti.bounded(1, X, np.inf))

        cs.append(c)
        Xs.append(X)

    log_mu = opti.variable()
    log_h_mu = opti.variable()
    log_nu   = opti.variable()
    log_b0 = opti.variable()
    log_d0 = opti.variable()
    log_d_d0 = opti.variable()
    log_b1 = opti.variable()
    log_d1 = opti.variable()

    opti.subject_to(opti.bounded(np.log(1e-6), log_mu, np.log(1e-1)))
    opti.subject_to(opti.bounded(np.log(1e-6), log_h_mu, np.log(1e-1)))
    opti.subject_to(opti.bounded(np.log(1e-6), log_nu, np.log(1e-1)))
    opti.subject_to(opti.bounded(np.log(1e-6), log_b0, np.log(1e-1)))
    opti.subject_to(opti.bounded(np.log(1e-6), log_d0, np.log(1e-1)))
    opti.subject_to(opti.bounded(np.log(1e-6), log_d_d0, np.log(1e-1)))
    opti.subject_to(opti.bounded(np.log(1e-6), log_b1, np.log(1e-1)))
    opti.subject_to(opti.bounded(np.log(1e-6), log_d1, np.log(1e-1)))

    if perfect_initial:
        opti.set_initial(log_mu, np.log(true_params.mu))
        opti.set_initial(log_h_mu, np.log(true_params.h_mu))
        opti.set_initial(log_nu, np.log(true_params.nu))
        opti.set_initial(log_b0, np.log(true_params.b0))
        opti.set_initial(log_d0, np.log(true_params.d0))
        opti.set_initial(log_d_d0, np.log(true_params.d_d0))
        opti.set_initial(log_b1, np.log(true_params.b1))
        opti.set_initial(log_d1, np.log(true_params.d1))


    params = [log_mu, log_h_mu, log_nu, log_b0, log_d0, log_d_d0, log_b1, log_d1]

    return optimize(Xs, cs, params, opti)


# Multiple shooting fitting states to a fixed control sequence using an ODE integrator
# Controls c_data are provided externally (length N)
def multiple_shooting_fixed_c_test_true(sim, true_params, Xs_true):
    times = sim.type.meas_times
    N = len(times) - 1
    print(N)

    # Create Opti instance
    opti = ca.Opti()

    # Parameter: control vector
    cs = []
    Xs = []
    params = []

    # Decision variables: state nodes (2 x (N+1))
    for dose_sched, meas, x0, x1 in zip(sim.type.doses, sim.data, Xs_true[0], Xs_true[1]):
        c = opti.parameter(N)
        opti.set_value(c, dose_sched)

        X = opti.parameter(2, N+1)
        opti.set_value(X[0, :], x0)
        opti.set_value(X[1, :], x1)
        # Initial condition
        # opti.subject_to(X[:, 0] == ca.DM(X0))

        # Constrain states to be nonnegative
        #opti.subject_to(opti.bounded(1, X, np.inf))

        cs.append(c)
        Xs.append(X)

    log_mu = opti.parameter()
    log_h_mu = opti.parameter()
    log_nu   = opti.parameter()
    log_b0 = opti.parameter()
    log_d0 = opti.parameter()
    log_d_d0 = opti.parameter()
    log_b1 = opti.parameter()
    log_d1 = opti.parameter()

    opti.set_value(log_mu, np.log(true_params.mu))
    opti.set_value(log_h_mu, np.log(true_params.h_mu))
    opti.set_value(log_nu, np.log(true_params.nu))
    opti.set_value(log_b0, np.log(true_params.b0))
    opti.set_value(log_d0, np.log(true_params.d0))
    opti.set_value(log_d_d0, np.log(true_params.d_d0))
    opti.set_value(log_b1, np.log(true_params.b1))
    opti.set_value(log_d1, np.log(true_params.d1))

    params = [log_mu, log_h_mu, log_nu, log_b0, log_d0, log_d_d0, log_b1, log_d1]

    return optimize(Xs, cs, params, opti)

# Example usage
if __name__ == '__main__':
    import numpy as np

    sim_type = utils.MeasurementType(
        change_times = [k * 10 for k in range(20)],
        meas_times = [k * 10 for k in range(20 + 1)],
        doses = np.array([[1] * 10 + [0] * 10, 
                          [2] * 10 + [0] * 10,
                          [5] * 10 + [0] * 10,
                          [10] * 10 + [0] * 10,
                          [50] * 10 + [0] * 10,
                          [100] * 10 + [0] * 10,
                          ])
    )

    lin_param_default_no_h_nu = utils.LastYearParamSetLinear_no_h_nu(
        mu = 0.0004,
        h_mu = 0.00004,
        nu = 0.004,
        b0 = 0.05,
        d0 = 0.01,
        d_d0 = 0.08,
        b1 = 0.002,
        d1 = 0.001
    )
    #cnt_calc = utils.calc_meas_mat(sim_type, lin_param_default_no_h_nu, 10/11, np.array([1100])) 

    sim, true_hist = simulate(lin_param_default_no_h_nu, sim_type, 1000, 100, 0.05)

    print('=== true values ===')
    X_trajs, P_opt = multiple_shooting_fixed_c_test_true(sim, lin_param_default_no_h_nu, true_hist)
    print(f'P_opt: {P_opt}')
    print(f'exp(P_opt) {np.exp(list(P_opt))}')
    print("Fitted state trajectories:\n", X_trajs)
    fig = go.Figure()
    for X_traj, meas in zip(X_trajs, sim.data):
        fig.add_scatter(x=sim_type.meas_times, y=np.array(X_traj[0]) + np.array(X_traj[1]), mode='lines+markers')
        fig.add_scatter(x=sim_type.meas_times, y=meas, mode='lines+markers', line=dict(dash='dash'))

    fig.update_yaxes(type="log")
    fig.show()

    print('=== estimated values ===')
    X_trajs, P_opt = multiple_shooting_fixed_c(sim, True, lin_param_default_no_h_nu)
    print(f'P_opt: {P_opt}')
    print(f'exp(P_opt) {np.exp(list(P_opt))}')
    print("Fitted state trajectories:\n", X_trajs)
    fig = go.Figure()
    for X_traj, meas in zip(X_trajs, sim.data):
        fig.add_scatter(x=sim_type.meas_times, y=np.array(X_traj[0]) + np.array(X_traj[1]), mode='lines+markers')
        fig.add_scatter(x=sim_type.meas_times, y=meas, mode='lines+markers', line=dict(dash='dash'))

    fig.update_yaxes(type="log")
    fig.show()

