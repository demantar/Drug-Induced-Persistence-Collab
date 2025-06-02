# NOTE! Plotting code was written by ChatGPT
# The mathematics was written by BVM and adapted from the code from ChatGPT
import streamlit as st
import streamlit.components.v1 
import numpy as np
from scipy.integrate import solve_ivp
import scipy.integrate as integrate
import plotly.graph_objects as go
import math

st.set_page_config(layout="wide")

with open("load-mathjax.js", "r") as f:
    js = f.read()
    streamlit.components.v1.html(f"<script>{js}</script>", height=0)


# Title
st.title("Interactive model simulator")

st.latex(
    r"""
    f_0' = (\lambda_1 - \lambda_0)f_0^2 - (\lambda_1 - \lambda_0 + \mu + \nu) f_0 + \nu
    """
)

st.sidebar.header("Parameters")

# Sliders
log_nu = st.sidebar.slider('log(nu)', min_value=-5.0, max_value=0.0, step=0.02, value=math.log10(0.002))
log_mu = st.sidebar.slider('log(mu)', min_value=-5.0, max_value=0.0, step=0.02, value=math.log10(0.002))
log_b0 = st.sidebar.slider('log(b_0)', min_value=-4.0, max_value=0.0, step=0.02, value=math.log10(0.04))
log_d0 = st.sidebar.slider('log(d_0)', min_value=-4.0, max_value=0.0, step=0.02, value=math.log10(0.08))
log_b1 = st.sidebar.slider('log(b_1)', min_value=-4.0, max_value=0.0, step=0.02, value=math.log10(0.002))
log_d1 = st.sidebar.slider('log(d_1)', min_value=-4.0, max_value=0.0, step=0.02, value=math.log10(0.001))
t_max = st.sidebar.slider('Final Time', min_value=0.0, max_value=1000.0, step=5.0, value=300.0)
y0 = st.sidebar.slider('initial f0', min_value=0.0, max_value=1.0, step=0.05, value=1.0)


nu = math.pow(10, log_nu)
mu = math.pow(10, log_mu)
b0 = math.pow(10, log_b0)
b1 = math.pow(10, log_b1)
d0 = math.pow(10, log_d0)
d1 = math.pow(10, log_d1)
lambda0 = b0 - d0 
lambda1 = b1 - d1

# ODE definition
def f0_ode(t, y):
    return (lambda1 - lambda0) * y * y - (lambda1 - lambda0 + mu + nu) * y + nu

# Solve the ODE
t_eval = np.linspace(0, t_max, 500)
f0_sol = solve_ivp(f0_ode, (0, t_max), [y0], t_eval=t_eval)
f0_sol_y = f0_sol.y[0]
f0_sol_t = f0_sol.t

rho_sol_y = (lambda0 - lambda1) * f0_sol_y + lambda1
rho_sol_t = f0_sol.t

log_growth_y = integrate.cumulative_simpson(rho_sol_y, x=rho_sol_t) 
log_growth_t = rho_sol_t

death_rate = b0 * f0_sol.t + b1 * (1 - f0_sol.t)
# NOTE! this jumps to exponential sizes and then integrates so error might be a problem
log_size_w_death_y = np.log(integrate.cumulative_simpson(np.exp(log_growth_y) * death_rate[:-1], x=rho_sol_t[:-1]) + np.exp(log_growth_y)[:-1])
log_size_w_death_t = rho_sol_t

# --- Session state for frozen plots ---
if "frozen_plots" not in st.session_state:
    st.session_state.frozen_plots = []

# --- Button to freeze current plot ---
if st.sidebar.button("Freeze current plot"):
    st.session_state.frozen_plots.append({
        "f0_sol_t": f0_sol_t.copy(),
        "f0_sol_y": f0_sol_y.copy(),
        "rho_sol_t": rho_sol_t.copy(),
        "rho_sol_y": rho_sol_y.copy(),
        "log_growth_t": log_growth_t.copy(),
        "log_growth_y": log_growth_y.copy(),
        "log_size_w_death_t": log_size_w_death_t.copy(),
        "log_size_w_death_y": log_size_w_death_y.copy(),
        "label": ''
    })

col1, col2 = st.columns(2)

# Plot using Plotly
fig_f0 = go.Figure()
fig_f0.add_trace(go.Scatter(x=f0_sol_t, y=f0_sol_y, mode='lines', name='f_0(t)'))
fig_f0.update_layout(yaxis_range=[0,1])
fig_f0.update_layout(
    title='state proportions',
    xaxis_title='$t$',
    yaxis_title=r'$f_0(t)$',
    template='plotly_white'
)

# Add frozen plots
for trace in st.session_state.frozen_plots:
    fig_f0.add_trace(go.Scatter(
        x=trace["f0_sol_t"], y=trace["f0_sol_y"],
        mode='lines',
        name=f"Frozen: {trace['label']}",
        line=dict(dash='dot', width=2)
    ))


col1.plotly_chart(fig_f0, use_container_width=True)

fig_rho = go.Figure()
fig_rho.add_trace(go.Scatter(x=rho_sol_t, y=rho_sol_y, mode='lines', name='rho(t)'))
fig_rho.update_layout()
fig_rho.update_layout(
    title='growth rate',
    xaxis_title='$t$',
    yaxis_title=r'$\rho(t)$',
    template='plotly_white'
)

for trace in st.session_state.frozen_plots:
    fig_rho.add_trace(go.Scatter(
        x=trace["rho_sol_t"], y=trace["rho_sol_y"],
        mode='lines',
        name=f"Frozen: {trace['label']}",
        line=dict(dash='dot', width=2)
    ))

col2.plotly_chart(fig_rho, use_container_width=True)

# Plot using Plotly
fig_log_growth = go.Figure()
fig_log_growth.add_trace(go.Scatter(x=log_growth_t, y=log_growth_y, mode='lines', name='f_0(t)'))
fig_log_growth.update_layout(
    title='logarithm of total growth',
    xaxis_title='$t$',
    yaxis_title=r'$\log(n(t)/n(0))$',
    template='plotly_white'
)

# Add frozen plots
for trace in st.session_state.frozen_plots:
    fig_log_growth.add_trace(go.Scatter(
        x=trace["log_growth_t"], y=trace["log_growth_y"],
        mode='lines',
        name=f"Frozen: {trace['label']}",
        line=dict(dash='dot', width=2)
    ))


col2.plotly_chart(fig_log_growth, use_container_width=True)

# Plot using Plotly
fig_size_w_death = go.Figure()
fig_size_w_death.add_trace(go.Scatter(x=log_size_w_death_t, y=log_size_w_death_y, mode='lines', name='f_0(t)'))
fig_size_w_death.update_layout(
    title='logarithm of total size with dead',
    xaxis_title='$t$',
    yaxis_title=r'$\log(n_{\text{tot}}(t)/n(0))$',
    template='plotly_white'
)

# Add frozen plots
for trace in st.session_state.frozen_plots:
    fig_size_w_death.add_trace(go.Scatter(
        x=trace["log_size_w_death_t"], y=trace["log_size_w_death_y"],
        mode='lines',
        name=f"Frozen: {trace['label']}",
        line=dict(dash='dot', width=2)
    ))


col1.plotly_chart(fig_size_w_death, use_container_width=True)
