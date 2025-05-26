import pandas as pd
import plotly.express as px
import numpy as np

df = pd.read_csv('param_est_3.csv')

df = df.pivot(index=["Run", "Type"], columns="Parameter", values="Value").reset_index()

print(df.to_string())

# 2. Compute new derived parameter(s)
df['abs(lambda0)'] = np.abs(df['b0'] - df['d0'])
df['abs(lambda1)'] = np.abs(df['b1'] - df['d1'])
df['abs(h_nu)'] = np.abs(df['h_nu'])


df_long = df.melt(
    id_vars=["Run", "Type"],
    value_vars=["mu", "h_mu", "nu", "abs(h_nu)", "abs(lambda0)", "abs(lambda1)"],
    var_name="Parameter",
    value_name="Value"
)

print(df_long)

fig = px.box(
    df_long,
    x="Parameter",
    y="Value",
    points="all",
    color="Type",
    hover_data=["Run"]
)

fig.update_yaxes(type="log")
fig.update_layout(title="Parameter Estimates vs True Values", height=400)

fig.show()


