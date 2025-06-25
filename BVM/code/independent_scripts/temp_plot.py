import numpy as np
import plotly.express as px


D = [0.01, 0.1, 1, 10, 100] 

c = np.power(10, np.linspace(np.log10(min(D)), np.log10(max(D)), 100))
d_ratio = c / (c + 1)

print(d_ratio)

fig = px.line(x=c, y=d_ratio, log_x=True)
for d in D:
    fig.add_vline(x = d)

fig.show()

