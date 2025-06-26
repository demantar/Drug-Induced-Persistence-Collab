from base64 import b16decode
import pandas as pd
import plotly.express as px
import model_utils as utils
import numpy as np

filename = '../results/simulation_experiments/12jun/param_est_9.csv'

df = pd.read_csv(filename)
df = df.pivot(index=["Run", "Type"], columns="Parameter", values="Value").reset_index()

print(df.head())

# Convert to Python native types
df["Run"] = df["Run"].astype(str)
df["Type"] = df["Type"].astype(str)

df['lambda0_0'] = df['b0'] - df['d0']
df['lambda1_0'] = df['b1'] - df['d1']
df['lambda1_0 - nu'] = df['lambda1_0'] - df['nu']

mask = df["Type"] == 'True'
df.loc[mask, 'f0_init'] = df.loc[mask, 'f0']


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

    params_c= utils.get_fund_param_set(params, 0)

    f, rho = utils.calc_equilib(params_c)
    f0_eq_c = f[0] 
    df.loc[idx, 'f0_eq_c'] = f0_eq_c
    df.loc[idx, 'rho_eq_c'] = rho
    lambda0_c = params_c.b0 - params_c.d0
    df.loc[idx, 'lambda0_c'] = lambda0_c
    lambda1_c = params_c.b1 - params_c.d1
    df.loc[idx, 'lambda1_c'] = lambda1_c
    mu_c = params_c.mu
    df.loc[idx, 'mu_c'] = mu_c
    nu_c = params_c.nu
    df.loc[idx, 'nu_c'] = nu_c

    alpha_c = lambda0_c - mu_c
    df.loc[idx, 'alpha_c'] = alpha_c

    beta_c = lambda1_c - nu_c
    df.loc[idx, 'beta_c'] = beta_c

    A = np.matrix([[alpha_c, mu_c], [nu_c, beta_c]])
    v = np.matrix([[f[0], f[1]]])
    v_ = v @ A
    print(f'f: {f}, rho: {rho}, v: {v}, v_: {v_}, v_/v: {v_/v}')
    print(A)
    print(utils.inf_gen_mat(utils.get_fund_param_set(params, 10)))


df['f1_eq_c'] = 1 - df['f0_eq_c']
df['denom_c'] = 1 / (1 - df['f0_eq_c'])
df['mlnr_c'] = df['mu_c'] / (df['mu_c'] - df['beta_c'] + df['rho_eq_c']) 
df['nlmr_c'] = df['nu_c'] / (df['nu_c'] - (df['alpha_c'])  + df['rho_eq_c']) 

df['mlnr_0_approx'] = df['mu_c'] / (df['mu_c'] - df['beta_c'] + df['rho_eq_c']) 

df['f1_init'] = 1 - df['f0_init']


comp_param1 = 'f1_init'
comp_param2 = 'mu'

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
    print(f'p1: {row[comp_param1]}')
    errors2.append(float(row[comp_param2] / true2))
    values2.append(float(row[comp_param2]))
    print(f'p2: {row[comp_param2]}')

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

#fig.show()
