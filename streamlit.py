import streamlit as st
import numpy as np
import plotly.graph_objects as go

st.title("Interactive K-Means Clustering")
st.write("Enter the coordinates of new points manually below!")

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

# Ensure axis range covers data
fig.update_layout(
    title="K-Means Clustering", 
    xaxis_title="X-axis", 
    yaxis_title="Y-axis",
    xaxis=dict(range=[0, 100]),  # Manually setting the axis range
    yaxis=dict(range=[0, 100])
)

# --- Handle New Point Input ---
x_input = st.number_input("Enter X-coordinate for new point:", min_value=0, max_value=100, value=50)
y_input = st.number_input("Enter Y-coordinate for new point:", min_value=0, max_value=100, value=50)

# Add point to clicked points list
if st.button("Add Point"):
    st.session_state.clicked_points.append((x_input, y_input))
    st.write(f"Added point at ({x_input}, {y_input})")
    st.rerun()  # Refresh UI

# --- Show the Plotly chart once here ---
st.plotly_chart(fig)

# --- Debugging Info ---
# Display data points and classes for debugging
st.write("Generated Data Points:")
st.write(data[:10])  # Show first 10 points for debugging

st.write("Centroids:")
st.write(centroids)

st.write("Classes:")
st.write(classes)
