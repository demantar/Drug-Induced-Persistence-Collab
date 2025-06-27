# Optimal control for the linear case using Dymos
import numpy as np
import openmdao.api as om
import dymos as dm

mu_0 = 0.0004
h_mu = 0.0004
nu_0 = 0.004
h_nu = -0.0004
b0 = 0.04
d0_0 = 0.0
d_d0 = 0.08
b1 = 0.001
d1 = 0.0

# case to use (1: mu linear, 2: nu linear, 3: both linear)
case = 3 

class MyODE(om.ExplicitComponent):
    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']

        self.add_input('f0', shape=(nn,))
        self.add_input('c', shape=(nn,))

        self.add_output('f0_dot', shape=(nn,))
        self.add_output('cost_rate', shape=(nn,))

        self.declare_partials(of='*', wrt='*', method='cs')

    def compute(self, inputs, outputs):
        f0 = inputs['f0']
        c = inputs['c']

        # Code to calculate how the parameters depend on f0 and c
        if case in [1, 3]:
            mu = mu_0 + c * h_mu 
        else:
            mu = mu_0

        if case in [2, 3]:
            nu = np.maximum(0, nu_0 + c * h_nu)
        else: 
            nu = nu_0

        lambda0 = b0 - (d0_0 + d_d0 * c / (1 + c))
        lambda1 = b1 - d1

        term1 = (lambda1 - lambda0) * f0**2
        term2 = -(lambda1 - lambda0 + mu + nu) * f0
        term3 = nu

        outputs['f0_dot'] = term1 + term2 + term3
        outputs['cost_rate'] = f0 * lambda0 + (1 - f0) * lambda1


# Set up the problem
prob = om.Problem(model=om.Group())

prob.driver = om.pyOptSparseDriver(print_results=True)
prob.driver.options['optimizer'] = 'IPOPT'
prob.driver.opt_settings.update({
    'print_level': 5,        
    'max_iter': 1000,        
    'tol': 1e-8,             
})

traj = dm.Trajectory()
phase = dm.Phase(ode_class=MyODE, transcription=dm.Radau(num_segments=40, order=4))

traj.add_phase('phase0', phase)
prob.model.add_subsystem('traj', traj)

phase.set_time_options(fix_initial=True, fix_duration=True, duration_bounds=(0.0, 1200.0))

phase.add_state('f0', rate_source='f0_dot', fix_initial=True, fix_final=False)
phase.set_state_options('f0', lower=1e-6, upper=1 - 1e-6)

phase.add_state('J', fix_initial=True, fix_final=False,
                    rate_source='cost_rate')

phase.add_control('c', lower=0.00, upper=10.0)

phase.add_objective('J', loc='final')

prob.setup()

prob.set_val('traj.phase0.t_initial', 0.0)
prob.set_val('traj.phase0.t_duration', 1200.0)

# calculate the equilibrium ratio (the initial f0)
lambda0 = b0 - d0_0
lambda1 = b1 - d1
A = np.matrix([[lambda0 - mu_0, mu_0], [nu_0, lambda1 - nu_0]])

eigvals, eigvecs = np.linalg.eig(A.T)
dominant_idx = np.argmax(eigvals) 
dominant_left_eigvec = eigvecs[:, dominant_idx]
dominant_left_eigvec = dominant_left_eigvec / np.sum(dominant_left_eigvec)
f0_initial = dominant_left_eigvec[0]

J_initial = 0   

prob.set_val('traj.phase0.states:f0', f0_initial)
prob.set_val('traj.phase0.states:J', J_initial)
prob.set_val('traj.phase0.controls:c', phase.interp(ys=[0.0, 0.0], nodes='control_input'))

dm.run_problem(prob, simulate=True)

from dymos.examples.plotting import plot_results

# Plot the results as a sanity check
sol = om.CaseReader(prob.get_outputs_dir() / 'dymos_solution.db').get_case('final')
sim_prob = prob.model.traj.sim_prob
sim = om.CaseReader(sim_prob.get_outputs_dir() / 'dymos_simulation.db').get_case('final')
fig, axes = plot_results([('traj.phase0.timeseries.time',
                        'traj.phase0.timeseries.f0',
                        'time ',
                        '$f_0$'),
                       ('traj.phase0.timeseries.time',
                        'traj.phase0.timeseries.c',
                        'time ',
                        '$c$'),
                       ('traj.phase0.timeseries.time',
                        'traj.phase0.timeseries.J',
                        'time ',
                        '$J$'),
                      ],
                      title='Test',
                      p_sol=sol, p_sim=sim)

import matplotlib.pyplot as plt
axes[0].set_xlim(0, 1200)
plt.show()

# Save important results
with open(f'linear{'I' * case}.txt', 'w') as f:
    print(f'times: {sim.outputs['traj.phase0.timeseries.time']}', file=f)
    print(f'phi: {sim.outputs['traj.phase0.timeseries.c']}', file=f)
    print(f'f0: {sim.outputs['traj.phase0.timeseries.f0']}', file=f)
    print(f'J: {sim.outputs['traj.phase0.timeseries.J']}', file=f)
