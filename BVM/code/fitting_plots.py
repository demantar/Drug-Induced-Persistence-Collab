# Several functions to plot the result of fits
import model_utils as utils
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from fitting import *
import itertools

# function to plot the logarithm of the size of the simulation,
# the deterministic simplification given true parameter def_params, 
# and the deterministic simplficiation given param estimate params
# It is good for testing and debugging parameter estimation
# Function partially written by ChatGPT
# WARNING: correct f0_init should be supplied for correct plot
def plot_params_fit_log_growth_pulsed(sim, params, def_params, f0_init=1.0):
    # Create figure
    fig = go.Figure()

    colors = px.colors.qualitative.Plotly

    log_growth_calc_true = np.log(utils.calc_meas_mat(sim.type, def_params, f0_init, 
                                                      np.ones(len(sim.type.doses))).data)
    log_growth_calc_fitted = np.log(utils.calc_meas_mat(sim.type, params, f0_init, 
                                                      np.ones(len(sim.type.doses))).data)

    for i, doses, counts in zip(itertools.count(), sim.type.doses, sim.data):

        color = colors[i % len(colors)]

        fig.add_trace(go.Scatter(
            x=sim.type.meas_times, y=np.log(counts/counts[0]),
            mode='lines',
            name=f'sim {i}',
            line=dict(color=color, dash='solid')
        ))

        fig.add_trace(go.Scatter(
            x=sim.type.meas_times, y=log_growth_calc_fitted[i],
            mode='lines',
            name=f'calc fitted {i}',
            line=dict(color=color, dash='dash')  
        ))

        fig.add_trace(go.Scatter(
            x=sim.type.meas_times, y=log_growth_calc_true[i],
            mode='lines',
            name=f'calc true {i}',
            line=dict(color=color, dash='dashdot') 
        ))

    # Update layout
    fig.update_layout(
        title='10 Pairs of Functions of Time',
        xaxis_title='Time',
        yaxis_title='Function Value',
        legend_title='Functions',
        template='plotly_dark', 
        height=600
    )

    fig.show()

# similar to last function but plots difference
def plot_params_fit_log_growth_pulsed_diff(sim, params, def_params, f0_init=1.0, 
                                           grouped_by_dose=True, show_individual_in_group=True):
    # Create figure
    fig = go.Figure()

    colors = px.colors.qualitative.Plotly

    log_growth_calc_true = np.log(utils.calc_meas_mat(sim.type, def_params, f0_init, 
                                                      np.ones(len(sim.type.doses))).data)
    log_growth_calc_fitted = np.log(utils.calc_meas_mat(sim.type, params, f0_init, 
                                                      np.ones(len(sim.type.doses))).data)

    if not grouped_by_dose:
        for i, doses, counts in zip(itertools.count(), sim.type.doses, sim.data):

            color = colors[i % len(colors)]

            fig.add_trace(go.Scatter(
                x=sim.type.meas_times, y=np.log(counts/counts[0]) - log_growth_calc_true[i],
                mode='lines',
                name=f'sim - true {i}',
                line=dict(color=color, dash='solid')
            ))

            fig.add_trace(go.Scatter(
                x=sim.type.meas_times, y=log_growth_calc_fitted[i] - log_growth_calc_true[i],
                mode='lines',
                name=f'calc fitted - calc_true {i}',
                line=dict(color=color, dash='dash')  
            ))
    else:
        fit_true_by_dose = {}
        sim_true_by_dose = {}
        for doses in sim.type.doses:
            sim_true_by_dose[tuple(doses)] = []
        for i, doses, counts in zip(itertools.count(), sim.type.doses, sim.data):
            d_tup = tuple(doses)
            fit_true_by_dose[d_tup] = log_growth_calc_fitted[i] - log_growth_calc_true[i]
            sim_true_by_dose[d_tup].append(np.log(counts/counts[0]) - log_growth_calc_true[i])

        for i, doses in enumerate(fit_true_by_dose):
            color = colors[i % len(colors)]
            fig.add_trace(go.Scatter(
                x=sim.type.meas_times, y=np.mean(sim_true_by_dose[doses], axis=0),
                mode='lines',
                name=f'sim - true {doses}',
                legendgroup=f'doses: {doses}',
                line=dict(color=color, dash='solid', width=1)
            ))

            if show_individual_in_group:
                for sim_true in sim_true_by_dose[doses]:
                    fig.add_trace(go.Scatter(
                        x=sim.type.meas_times, y=sim_true,
                        mode='lines',
                        name=f'sim - true {doses}',
                        legendgroup=f'doses: {doses}',
                        line=dict(color=color, dash='solid', width=0.2)
                    ))


            fig.add_trace(go.Scatter(
                x=sim.type.meas_times, y=fit_true_by_dose[doses],
                mode='lines',
                name=f'calc fitted - calc_true {doses}',
                legendgroup=f'doses: {doses}',
                line=dict(color=color, dash='dash')  
            ))




    # Update layout
    fig.update_layout(
        title='10 Pairs of Functions of Time',
        xaxis_title='Time',
        yaxis_title='Function Value',
        legend_title='Functions',
        template='plotly_dark', 
        height=600
    )

    fig.show()

# another testing / debugging function to test how the estimate for f0
# calculated from the estimated parameters compares to the deterministic
# estimation of f0 given the true parameters
def plot_params_fit_f0_pulsed(sim, params, def_params, f0_init=1.0): 
    # Create figure
    fig = go.Figure()

    colors = px.colors.qualitative.Plotly

    log_growth_calc_true = np.log(utils.calc_meas_mat(sim.type, def_params, f0_init, 
                                                      np.ones(len(sim.type.doses))).data)
    log_growth_calc_fitted = np.log(utils.calc_meas_mat(sim.type, params, f0_init, 
                                                      np.ones(len(sim.type.doses))).data)

    for i, doses, counts in zip(itertools.count(), sim.type.doses, sim.data):
        color = colors[i % len(colors)]

        switching_times = np.array(sim.type.change_times)
        values = np.array([0] + list(doses))

        c_t = lambda t: values[np.searchsorted(switching_times, t, side='right')]
        f0 = utils.sol_f0(params, c_t, sim.type.meas_times, 10/11)

        fig.add_trace(go.Scatter(
            x=sim.type.meas_times, y=f0,
            mode='lines',
            name=f'f_0 calc fit {i}',
            line=dict(color=color, dash='dash') 
        ))

        f0 = utils.sol_f0(def_params, c_t, sim.type.meas_times, 10/11)

        fig.add_trace(go.Scatter(
            x=sim.type.meas_times, y=f0,
            mode='lines',
            name=f'f_0 calc def {i}',
            line=dict(color=color, dash='dot') 
        ))

    # Update layout
    fig.update_layout(
        title='10 Pairs of Functions of Time',
        xaxis_title='Time',
        yaxis_title='Function Value',
        legend_title='Functions',
        template='plotly_dark', 
        height=600
    )

    fig.show()
