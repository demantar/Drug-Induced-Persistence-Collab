# File for plotting the fitting plots resulting from the simulations
# NOTE: a lot of the plotting is written by ChatGPT (under superivsion)
import pandas as pd
import chart_studio.plotly as py
import chart_studio.tools as tls
import plotly.graph_objects as go
import os

# whether to upload the plots into the cloud or simply display them
display_method = "browser"
save_dir = "figure_pngs"
assert display_method in ["browser", "save to png", "upload to plotly"]

if display_method == "uploaed to plotly":
    tls.set_credentials_file(username='', api_key='') # change these to use cloud

# names of csv files to plot
filenames = [f"param_est_{i}.csv" for i in range(7, 7 + 1)]

# corresponding titles of the plots
titles = [
    "michaelis medten function",
]

# functiion that returns a single plot
def get_plot(filename, msg):
    # Read and transform
    df = pd.read_csv(filename)
    df = df.pivot(index=["Run", "Type"], columns="Parameter", values="Value").reset_index()

    # Derived parameters
    df['abs(lambda0)'] = (df['b0'] - df['d0']).abs()
    df['abs(lambda1)'] = (df['b1'] - df['d1'])#.abs()
    df['abs(lambda1 - nu)'] = (df['b1'] - df['d1'] - df['nu']).abs()
    #df['abs(h_nu)'] = df['h_nu'].abs()
    df['f0/100'] = df['f0_init'] / 100

    # Melt for long format
    df_long = df.melt(
        id_vars=["Run", "Type"],
        #value_vars=["mu", "h_mu", "nu", "abs(lambda0)", "abs(lambda1)", "d_d0", "f0/100", "abs(lambda1 - nu)"],
        value_vars=["mu", "d_mu", "e_mu", "nu", "abs(lambda0)", "abs(lambda1)", "d_d0", "f0/100", "abs(lambda1 - nu)"],
        var_name="Parameter",
        value_name="Value"
    )

    # Convert to Python native types
    df_long["Run"] = df_long["Run"].astype(str)
    df_long["Type"] = df_long["Type"].astype(str)
    df_long["Parameter"] = df_long["Parameter"].astype(str)
    df_long["Value"] = df_long["Value"].astype(float)

    # Build figure manually (instead of using px.box)
    fig = go.Figure()

    # display names (for latex)
    var_to_disp_name = {
        'mu': r'$\mu$',
        #'h_mu': r'$h_\mu$',
        'd_mu': r'$\Delta \mu$',
        'e_mu': r'$E_\mu$',
        'nu': r'$\nu$',
        'abs(lambda0)': r'$\lvert \lambda_0 \rvert$',
        'abs(lambda1)': r'$\lvert \lambda_1 \rvert$',
        'd_d0': r'$\Delta d_0$',
        'f0/100': r'$f_0/100$',
        'abs(lambda1 - nu)': r'$\lvert \lambda_1  - \nu\rvert$'
    }
    
    for param in df_long["Parameter"].unique():
        if param in var_to_disp_name:
            disp_name = var_to_disp_name[param]
        else:
            disp_name = param
        for t in df_long["Type"].unique():
            filtered = df_long[(df_long["Parameter"] == param) & (df_long["Type"] == t)]
            fig.add_trace(go.Box(
                y=filtered["Value"].tolist(),
                x=[disp_name] * len(filtered),
                boxpoints="all",
                jitter=0.5,
                pointpos=-1.8,
                marker=dict(opacity=0.5),
                name=f"{param} - {t}",
                hovertext=filtered["Run"].tolist(),
                hoverinfo="text+y"
            ))

    fig.update_layout(
        title=f"{msg} ({filename})",
        height=400,
        xaxis_title="Parameter",
        yaxis_title="Value",
        showlegend=False,
        template="plotly"  # ensure compatibility
    )

    logarithmic_axis = True
    
    if logarithmic_axis:
        fig.update_yaxes(type="log", range=[-9, 2])
    else:
        fig.update_yaxes(range=[-9, 2])

    return fig

if display_method == "save to png":
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

for filename, title in zip(filenames, titles):
    fig = get_plot(filename, title)
    if display_method == "upload to plotly":
        py.plot(fig, filename=title, auto_open=False)
    elif display_method == "browser": 
        fig.show()
    elif display_method == "save to png":
        fig.write_image(f"{save_dir}/{title}.png")
    else:
        raise Exception("invalid/no display method")


