import pandas as pd
import plotly.express as px
import numpy as np
import chart_studio.plotly as py
import chart_studio.tools as tls
import plotly.graph_objects as go
import sys

tls.set_credentials_file(username='benediktvm', api_key='srhhXtHgU7EhSvi9Vslf')

if len(sys.argv) > 1:
    file = sys.argv[1]
else:
    file = 'param_est_3.csv'

if len(sys.argv) > 2:
    title = sys.argv[2]
else:
    title = "Parameter Estimates"

def get_plot(filename, msg):
    """
    df = pd.read_csv(filename)

    df = df.pivot(index=["Run", "Type"], columns="Parameter", values="Value").reset_index()


    # 2. Compute new derived parameter(s)
    df['abs(lambda0)'] = np.abs(df['b0'] - df['d0'])
    df['abs(lambda1)'] = np.abs(df['b1'] - df['d1'])
    df['abs(h_nu)'] = np.abs(df['h_nu'])

    df_long = df.melt(
        id_vars=["Run", "Type"],
        value_vars=["mu", "h_mu", "nu", "abs(h_nu)", "abs(lambda0)", "abs(lambda1)", "d_d0"],
        #value_vars=["mu", "h_mu", "nu", "abs(lambda0)", "abs(lambda1)"],
        var_name="Parameter",
        value_name="Value"
    )

    fig = px.box(
        df_long,
        x="Parameter",
        y="Value",
        points="all",
        color="Type",
        hover_data=["Run"]
    )

    fig.update_layout(title=f"{title} ({msg})", height=400)
    fig.update_yaxes(type="log", range=[-7, -0.5])

    return fig
    """

    # Read and transform
    df = pd.read_csv(filename)
    df = df.pivot(index=["Run", "Type"], columns="Parameter", values="Value").reset_index()

    # Derived parameters
    df['abs(lambda0)'] = (df['b0'] - df['d0']).abs()
    df['abs(lambda1)'] = (df['b1'] - df['d1']).abs()
    df['abs(h_nu)'] = df['h_nu'].abs()

    # Melt for long format
    df_long = df.melt(
        id_vars=["Run", "Type"],
        value_vars=["mu", "h_mu", "nu", "abs(h_nu)", "abs(lambda0)", "abs(lambda1)", "d_d0"],
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

    for param in df_long["Parameter"].unique():
        for t in df_long["Type"].unique():
            filtered = df_long[(df_long["Parameter"] == param) & (df_long["Type"] == t)]
            fig.add_trace(go.Box(
                y=filtered["Value"].tolist(),
                x=[param] * len(filtered),
                boxpoints="all",
                jitter=0.5,
                pointpos=-1.8,
                marker=dict(opacity=0.5),
                name=f"{param} - {t}",
                hovertext=filtered["Run"].tolist(),
                hoverinfo="text+y"
            ))

    fig.update_layout(
        title=f"{title} ({msg})",
        height=400,
        xaxis_title="Parameter",
        yaxis_title="Value",
        showlegend=False,
        template="plotly"  # ensure compatibility
    )

    fig.update_yaxes(type="log", range=[-7, -0.5])

    return fig

filenames = [f"param_est_{i}.csv" for i in range(1, 13)]

titles = [
     "normal pulsed (longer) - normal-rms, 2 replicates, every 30 sec",
     "normal fixed (longer) - normal-rms, 2 replicates, every 30 sec",
     "normal mixed (longer) - normal-rms, 2 replicates, every 30 sec",
     "normal pulsed (longer) - normal-rms, 2 replicates, every 10 sec",
     "normal fixed (longer) - normal-rms, 2 replicates, every 10 sec",
     "normal mixed (longer) - normal-rms, 2 replicates, every 10 sec",
     "normal pulsed (longer) - normal-rms, 6 replicates, every 30 sec",
     "normal fixed (longer) - normal-rms, 6 replicates, every 30 sec",
     "normal mixed (longer) - normal-rms, 6 replicates, every 30 sec",
     "normal pulsed (longer) - normal-rms, 6 replicates, every 10 sec",
     "normal fixed (longer) - normal-rms, 6 replicates, every 10 sec",
     "normal mixed (longer) - normal-rms, 6 replicates, every 10 sec"
]

for filename, title in zip(filenames, titles):
    fig = get_plot(filename, title)
    py.plot(fig, filename=title, auto_open=False)
    #fig.show()


