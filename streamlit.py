import streamlit as st
import numpy as np
import plotly.graph_objects as go
from streamlit_plotly_events import plotly_events

st.title("Interactive K-Means Clustering")
st.write("Click on the plot to add a new point!")

# --- Store clicked points ---
if "clicked_points" not in st.session_state:
    st.session_state.clicked_points = []

# --- Generate Random Dataset ---
def generate_random_dataset(k, n, spread, seed=42):
    np.random.seed(seed)
    clusters = []
    centers = np.random.uniform(0, 100, size=(k, 2))
    for center in centers:
        points = center + spread * np.random.randn(n, 2)
        clusters.append(points)
    return np.vstack(clusters)

data = generate_random_dataset(k=4, n=300, spread=10)

# --- K-Means Clustering ---
def k_means_clustering(data, k, iterations=10):
    centroids = data[np.random.choice(len(data), k, replace=False)]
    for _ in range(iterations):
        classes = {tuple(c): [] for c in centroids}
        for point in data:
            closest_centroid = min(centroids, key=lambda c: np.linalg.norm(point - c))
            classes[tuple(closest_centroid)].append(point)
        centroids = [np.mean(points, axis=0) if points else centroid for centroid, points in classes.items()]
    return np.array(centroids), classes

k = 4
centroids, classes = k_means_clustering(data, k)

# --- Define colors ---
colors = ['red', 'blue', 'green', 'cyan', 'magenta', 'yellow']

# --- Create Plotly Figure ---
fig = go.Figure()

# Add clusters
for i, (centroid, points) in enumerate(classes.items()):
    points = np.array(points)
    fig.add_trace(go.Scatter(
        x=points[:, 0], y=points[:, 1], mode='markers',
        name=f'Cluster {i+1}', marker=dict(color=colors[i % len(colors)])
    ))

# Add centroids
fig.add_trace(go.Scatter(
    x=centroids[:, 0], y=centroids[:, 1], mode='markers', name='Centroids',
    marker=dict(color='black', symbol='x', size=10)
))

# Add dynamically clicked points
if st.session_state.clicked_points:
    clicked_x, clicked_y = zip(*st.session_state.clicked_points)
    fig.add_trace(go.Scatter(
        x=clicked_x, y=clicked_y, mode='markers', name='Clicked Points',
        marker=dict(color='orange', size=12, symbol="circle-open")
    ))

fig.update_layout(title="K-Means Clustering", xaxis_title="X-axis", yaxis_title="Y-axis")

# --- Capture Click Events ---
click_data = plotly_events(fig, click_event=True)  # Captures clicks

# --- Handle Click Event ---
if click_data:
    new_x, new_y = click_data[0]["x"], click_data[0]["y"]
    if (new_x, new_y) not in st.session_state.clicked_points:  # Avoid duplicate points
        st.session_state.clicked_points.append((new_x, new_y))
        st.rerun()  # Refresh UI
