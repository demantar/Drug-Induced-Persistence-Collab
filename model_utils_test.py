import model_utils as mu
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

# Default parameters from last years article
lin_param_default = mu.LastYearParamSetLinear(
    mu = 0.0004,
    h_mu = 0.0004,
    nu = 0.004,
    h_nu = -0.0004,
    lambda0 = 0.04,
    d_lambda0 = -0.08,
    lambda1 = 0.001
)
hs_param_default = mu.LastYearParamSetHeaviside(
    mu = 0.0004,
    d_mu = 0.004,
    nu = 0.004,
    d_nu = -0.003,
    lambda0 = 0.04,
    d_lambda0 = -0.08,
    lambda1 = 0.001
)

constdose3 = lambda t : 3

times = np.linspace(0, 500, 300)

# how to calculate f0
f0 = mu.sol_f0(lin_param_default, constdose3, times, 1) 

# how to calculate rho
rho = mu.rho_c(f0, lin_param_default, constdose3(times))

log_growth = mu.log_growth(f0, lin_param_default, constdose3(times), times)

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
fig_log_growth .show()
