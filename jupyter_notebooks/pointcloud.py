import plotly.graph_objects as go
import numpy as np

points = np.loadtxt("lidar_snapshot.txt").reshape(-1, 4)

fig = go.Figure(data=[go.Scatter3d(
    x=points[:, 0],
    y=points[:, 1],
    z=points[:, 2],
    mode='markers',
    marker=dict(size=1, color=points[:, 2], colorscale='Viridis')
)])
fig.show()
