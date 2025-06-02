# A script to test the utilities in model_utils
# These tests are NOT comprihensive
import model_utils as utils
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

# Default parameters from last years article
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

constdose3 = lambda t : 3
times = np.linspace(0, 500, 300)

# how to calculate f0 via ODE
f0 = utils.sol_f0(lin_param_default, constdose3, times, 1) 

# how to calculate rho
rho = utils.rho_c(f0, lin_param_default, constdose3(times))

log_growth = utils.log_growth(f0, lin_param_default, constdose3, times)

print(rho.shape)
print(times.shape)
print(log_growth.shape)

df = pd.DataFrame({'f0':f0, 't':times})
fig_f0 = px.line(df, x='t', y='f0')
fig_f0.show()

df = pd.DataFrame({'rho':rho, 't':times})
fig_rho = px.line(df, x='t', y='rho')
fig_rho.show()

df = pd.DataFrame({'log growth':log_growth, 't':times})
fig_log_growth = px.line(df, x='t', y='log growth')
fig_log_growth.show()
