from base64 import b16decode
import pandas as pd
import plotly.express as px
import model_utils as utils

filename = 'param_est_9.csv'

df = pd.read_csv(filename)
df = df.pivot(index=["Run", "Type"], columns="Parameter", values="Value").reset_index()

print(df.head())

# Convert to Python native types
df["Run"] = df["Run"].astype(str)
df["Type"] = df["Type"].astype(str)

df['lambda0'] = df['b0'] - df['d0']
df['lambda1'] = df['b1'] - df['d1']
df['lambda1 - nu'] = df['lambda1'] - df['nu']


for idx, row in df.iterrows():
    # add columns iteratively (bad practice but simpler)
    params = utils.LastYearParamSetLinear_no_h_nu(
        mu = float(row['mu']), 
        h_mu = float(row['h_mu']),
        nu = float(row['nu']), 
        b0 = float(row['b0']),
        d0 = float(row['d0']),
        d_d0 = float(row['d_d0']),
        b1 = float(row['b1']),
        d1 = float(row['d1']),
    )

    df.loc[idx, 'f0_eq_10'] = utils.equilibf0(utils.get_fund_param_set(params, 10))
    df.loc[idx, 'rho_eq_10'] = utils.rho_c(df.loc[idx, 'f0_eq_10'], params, 10)

df['f1_eq_10'] = 1 - df['f0_eq_10']
df['denom10'] = 1 / (1 - df['f0_eq_10'])
df['mlnr10'] = df['mu'] / (df['mu'] - df['lambda1 - nu'] + df['rho_eq_10']) 
df['nlmr10'] = df['nu'] / (df['nu'] - (df['lambda0'] - df['mu'])  + df['rho_eq_10']) 

comp_param1 = 'f1_eq_10'
comp_param2 = 'mlnr10'

errors1 = []
true1 = df[df['Type'] == 'True'][comp_param1] 

errors2 = []
true2 = df[df['Type'] == 'True'][comp_param2] 
print(true1)
print(true2)
values1 = []
values2 = []

for idx, row in df.iterrows():
    if row['Type'] == 'True':
        continue
    errors1.append(float(row[comp_param1] / true1))
    values1.append(float(row[comp_param1]))
    print(row[comp_param1])
    errors2.append(float(row[comp_param2] / true2))
    values2.append(float(row[comp_param2]))
    print(row[comp_param2])

fig = px.scatter(x=errors1, y=errors2)

fig.update_layout(
    title=dict(
        text="Relative error comparison"
    ),
    xaxis=dict(
        title=dict(
            text=comp_param1
        )
    ),
    yaxis=dict(
        title=dict(
            text=comp_param2
        )
    ),
)

fig.show()

fig = px.scatter(x=values1, y=values2)

fig.update_layout(
    title=dict(
        text="values"
    ),
    xaxis=dict(
        title=dict(
            text=comp_param1
        )
    ),
    yaxis=dict(
        title=dict(
            text=comp_param2
        )
    ),
)

fig.show()
