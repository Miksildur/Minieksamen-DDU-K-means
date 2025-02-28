import streamlit as st
import numpy as np
import plotly.graph_objects as go

st.title("Interactive K-Means Clustering")
st.write("Enter a cluster center and generate a cluster of points around it!")

# --- Store cluster points ---
if "clicked_points" not in st.session_state:
    st.session_state.clicked_points = []

# Input for cluster center
st.sidebar.header("Enter Cluster Center")
x_center = st.sidebar.number_input("Cluster Center X", min_value=0, max_value=100, step=1)
y_center = st.sidebar.number_input("Cluster Center Y", min_value=0, max_value=100, step=1)

# Input for number of points and spread
num_points = st.sidebar.number_input("Number of Points", min_value=1, max_value=100, value=20, step=1)
spread = st.sidebar.number_input("Spread (Standard Deviation)", min_value=1, max_value=30, value=5, step=1)

# Button to generate points around the center
if st.sidebar.button("Generate Cluster"):
    # Generate random points around the given center
    points = np.random.normal(loc=[x_center, y_center], scale=spread, size=(num_points, 2))
    st.session_state.clicked_points.extend(points.tolist())
    st.write(f"Generated {num_points} points around the center ({x_center}, {y_center})")

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

# --- Create Plotly Figure ---
def create_plot():
    fig = go.Figure()

    # Add clicked points to plot
    if st.session_state.clicked_points:
        clicked_x, clicked_y = zip(*st.session_state.clicked_points)
        fig.add_trace(go.Scatter(
            x=clicked_x, y=clicked_y, mode='markers', name='Clicked Points',
            marker=dict(color='orange', size=12, symbol="circle-open")
        ))

    # Perform K-Means clustering if there are points
    if len(st.session_state.clicked_points) > 1:
        data = np.array(st.session_state.clicked_points)
        k = 4  # Number of clusters
        centroids, classes = k_means_clustering(data, k)

        # Add clusters
        colors = ['red', 'blue', 'green', 'cyan', 'magenta', 'yellow']
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

    # Update layout
    fig.update_layout(
        title="K-Means Clustering", 
        xaxis_title="X-axis", 
        yaxis_title="Y-axis",
        xaxis=dict(range=[0, 100]),  # Manually setting the axis range
        yaxis=dict(range=[0, 100])
    )

    return fig

# --- Display Plot ---
st.plotly_chart(create_plot(), use_container_width=True)
