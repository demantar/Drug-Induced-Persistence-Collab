import model_utils as utils
from simulate import *
import plotly.express as px
import plotly.graph_objects as go

sim_type = utils.MeasurementType(
        doses = [1,10,20,50,75,100],
        times = [0,10,20,30,40,50,60,70,80,90,100] 
)

sim_type_converted = utils.meas_type_to_meas_type_pulsed(sim_type)
print(sim_type_converted)

sim_type_pulsed = utils.MeasurementTypePulsed(
    change_times = [0, 20, 40, 60, 80, 100],
    meas_times = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
    doses = np.array([[1, 0, 1, 0, 1, 0], 
                      [5, 0, 5, 0, 5, 0],
                      [10, 0, 10, 0, 10, 0],
                      [20, 0, 20, 0, 20, 0], 
                      [50, 0, 50, 0, 50, 0],
                      [75, 0, 75, 0, 75, 0],
                      [100, 0, 100, 0, 100, 0]])
)

lin_param_default = utils.LastYearParamSetLinearBD(
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

sim_pulsed = simulate_pulsed(lin_param_default, sim_type_pulsed, 1000, 100, rel_meas_error=0.0)
print(sim_pulsed)

sim_converted = simulate_fixed(lin_param_default, sim_type, 1000, 100, rel_meas_error=0.0)
print(sim_converted)


fig = px.scatter(x = sim_type_pulsed.meas_times, y = sim_pulsed.data[0])
fig.show()

fig_pulsed = go.Figure()

sim_type = utils.MeasurementType(
        doses = [20],
        times = [0,10,20,30,40,50,60,70,80,90,100] 
)

sim_type_pulsed = utils.MeasurementTypePulsed(
    change_times = [0, 20, 40, 60, 80, 100],
    meas_times = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
    doses = np.array([[50, 0, 50, 0, 50, 0]])
)
for i in range(1000):
    sim1 = simulate_pulsed(lin_param_default, sim_type_pulsed, 1000, 100)
    fig_pulsed.add_trace(go.Scatter(
        x= sim_type.times,
        y= np.log(sim1.data[0]),
        mode='lines',
        line=dict(color='black', width=1),
        opacity=0.02  # Low opacity for blending
    ))

for i in range(6):
    sim1 = simulate_pulsed(lin_param_default, sim_type_pulsed, 1000, 100)
    fig_pulsed.add_trace(go.Scatter(
        x= sim_type.times,
        y= np.log(sim1.data[0]),
        mode='lines',
        line=dict(color='black', width=1),
        opacity=0.35  # Low opacity for blending
    ))

doses = np.array([0] + list(sim_type_pulsed.doses[0]))
c_t = lambda t: doses[np.searchsorted(sim_type_pulsed.change_times, t, side="right")]
eval_times = np.linspace(0, 100, 100)
f0 = utils.sol_f0_bd(lin_param_default, c_t, eval_times, 10 / 11)
#growth = utils.log_growth_bd(f0, lin_param_default, c_t, eval_times)
growth = np.log(utils.calc_meas_mat_bd(sim_type_pulsed, lin_param_default, 10/11, 1100).data[0])
fig_pulsed.add_trace(go.Scatter(
    x= sim_type.times,
    y= growth ,
    mode='lines',
    line=dict(color='red', width=1),
    opacity=1  # Low opacity for blending
))

sim1 = simulate_pulsed(lin_param_default, sim_type_pulsed, 1000, 100)
fig_pulsed.add_trace(go.Scatter(
    x= sim_type.times,
    y= np.log(sim1.data[0]),
    mode='lines',
    line=dict(color='black', width=1),
    opacity=1  # Low opacity for blending
))

fig_pulsed.update_layout(
    title='Time Series Overlap Density Plot',
    xaxis_title='Time',
    yaxis_title='Value',
    showlegend=False,
    plot_bgcolor='white'
)

fig_pulsed.show()

sim1 = simulate_pulsed(lin_param_default, sim_type_pulsed, 1000, 100)
print(sim1)


