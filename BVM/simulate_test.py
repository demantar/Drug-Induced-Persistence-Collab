import model_utils as utils
from simulate import *
import plotly.express as px
import plotly.graph_objects as go


# example of how to define a simulation type
sim_type = utils.MeasurementType(
    change_times = [0, 20, 40, 60, 80, 100],
    meas_times = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
    doses = np.array([[5, 0, 5, 0, 5, 0],
                      [10, 0, 10, 0, 10, 0],
                      [20, 0, 20, 0, 20, 0], 
                      [50, 0, 50, 0, 50, 0],
                      [75, 0, 75, 0, 75, 0],
                      [100, 0, 100, 0, 100, 0]])
)

# example of how to define a parameter set
lin_param_default = utils.LastYearParamSetLinear(
    mu = 0.0004,
    h_mu = 0.0004,
    nu = 0.004,
    h_nu = -0.0004,
    b0 = 0.04,
    d0 = 0.0,
    d_d0 = 0.08,
    b1 = 0.001,
    d1 = 0.0
)

# test one simulation
sim_pulsed = simulate(lin_param_default, sim_type, 1000, 100, rel_meas_error=0.05)
print(sim_pulsed)

# CHATGPT -- BEGIN
# Create the figure
fig = go.Figure()

# Add a line for each time series
for i, series in enumerate(sim_pulsed.data):
    fig.add_trace(go.Scatter(
        x=sim_type.meas_times,
        y=series,
        mode='lines',
        name=f"Series {i+1}"
    ))

# Customize layout
fig.update_layout(
    title="Measurements from one simulated experiment",
    xaxis_title="Time",
    yaxis_title="number of cells",
    template="plotly_white"
)

# Show the figure
fig.show()
# CHATGPT -- END

fig_pulsed = go.Figure()

# Note: the code for the following plots was partially written by ChatGPT
sim_type = utils.MeasurementType(
    change_times = [0, 20, 40, 60, 80, 100],
    meas_times = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
    doses = np.array([[50, 0, 50, 0, 50, 0]])
)
for i in range(1000):
    sim1 = simulate(lin_param_default, sim_type, 1000, 100, rel_meas_error=0.05)
    fig_pulsed.add_trace(go.Scatter(
        x= sim_type.meas_times,
        y= np.log(sim1.data[0]),
        mode='lines',
        line=dict(color='black', width=1),
        opacity=0.02  # Low opacity for blending
    ))

for i in range(6):
    sim1 = simulate(lin_param_default, sim_type, 1000, 100, rel_meas_error=0.05)
    fig_pulsed.add_trace(go.Scatter(
        x= sim_type.meas_times,
        y= np.log(sim1.data[0]),
        mode='lines',
        line=dict(color='black', width=1),
        opacity=0.35  # Low opacity for blending
    ))

doses = np.array([0] + list(sim_type.doses[0]))
c_t = lambda t: doses[np.searchsorted(sim_type.change_times, t, side="right")]
eval_times = np.linspace(0, 100, 100)
f0 = utils.sol_f0(lin_param_default, c_t, eval_times, 10 / 11)
growth = np.log(utils.calc_meas_mat(sim_type, lin_param_default, 10/11, 1100).data[0])
fig_pulsed.add_trace(go.Scatter(
    x= sim_type.meas_times,
    y= growth ,
    mode='lines',
    line=dict(color='red', width=1),
    opacity=1  # Low opacity for blending
))

sim1 = simulate(lin_param_default, sim_type, 1000, 100, rel_meas_error=0.05)
fig_pulsed.add_trace(go.Scatter(
    x= sim_type.meas_times,
    y= np.log(sim1.data[0]),
    mode='lines',
    line=dict(color='black', width=1),
    opacity=1  # Low opacity for blending
))

fig_pulsed.update_layout(
    title='Stochastic simulations vs deterministic model',
    xaxis_title='Time',
    yaxis_title='log(number of cells)',
    showlegend=False,
    plot_bgcolor='white'
)

fig_pulsed.show()
