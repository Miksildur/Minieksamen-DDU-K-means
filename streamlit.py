import streamlit as st
import numpy as np
import plotly.graph_objects as go

st.title("Interactive Animated K-Means Clustering")
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
    # Initialize centroids randomly from existing data
    centroids = data[np.random.choice(len(data), k, replace=False)]
    animations = []

    for _ in range(iterations):
        # Step 1: Assign points to nearest centroid
        distances = np.sqrt(((data - centroids[:, np.newaxis])**2).sum(axis=2))  # Shape (k, n)
        labels = np.argmin(distances, axis=0)  # Shape (n,)

        # Store current state for animation (centroids before update)
        animation_step = {
            'centroids': centroids.copy(),
            'labels': labels
        }
        animations.append(animation_step)

        # Step 2: Recalculate centroids
        new_centroids = np.array([data[labels == i].mean(axis=0) if np.sum(labels == i) > 0 else centroids[i] for i in range(k)])
        centroids = new_centroids

    # Calculate inertia (sum of squared distances)
    inertia = np.sum(np.min(distances, axis=0)**2)
    return animations, inertia

# --- Create Plotly Figure ---
def create_animation(animations):
    data = np.array(st.session_state.clicked_points)
    clicked_x = data[:, 0]
    clicked_y = data[:, 1]

    fig = go.Figure()

    # Add all points trace (initially gray)
    fig.add_trace(go.Scatter(
        x=clicked_x, y=clicked_y, mode='markers', name='Points',
        marker=dict(color='gray', size=5)
    ))

    # Add centroids trace (initially empty)
    fig.add_trace(go.Scatter(
        x=[], y=[], mode='markers', name='Centroids',
        marker=dict(color='black', symbol='x', size=10)
    ))

    frames = []
    for iteration, animation_step in enumerate(animations):
        centroids = animation_step['centroids']
        labels = animation_step['labels']

        # Generate colors for each point based on cluster
        colors = [f'hsl({(i * 360 / len(centroids)) % 360}, 100%, 50%)' for i in range(len(centroids))]
        point_colors = [colors[label] for label in labels]

        # Create frame with updated colors and centroids
        frame = go.Frame(
            data=[
                go.Scatter(x=clicked_x, y=clicked_y, marker=dict(color=point_colors)),
                go.Scatter(x=centroids[:, 0], y=centroids[:, 1])
            ],
            name=f'Frame {iteration}'
        )
        frames.append(frame)

    fig.frames = frames

    # Animation controls
    fig.update_layout(
        title="K-Means Clustering Animation",
        xaxis_title="X-axis",
        yaxis_title="Y-axis",
        xaxis=dict(range=[0, 100], autorange=False),
        yaxis=dict(range=[0, 100], autorange=False),
        updatemenus=[{
            'buttons': [
                {
                    'args': [None, {'frame': {'duration': 1000, 'redraw': True}, 'fromcurrent': True}],
                    'label': 'Play',
                    'method': 'animate'
                },
                {
                    'args': [[None], {'frame': {'duration': 0, 'redraw': True}, 'mode': 'immediate', 'transition': {'duration': 0}}],
                    'label': 'Pause',
                    'method': 'animate'
                }
            ],
            'direction': 'left',
            'pad': {'r': 10, 't': 87},
            'showactive': False,
            'type': 'buttons',
            'x': 0.1,
            'xanchor': 'right',
            'y': 0,
            'yanchor': 'top'
        }]
    )

    return fig

# --- Main Logic ---
if len(st.session_state.clicked_points) > 1:
    data = np.array(st.session_state.clicked_points)
    max_k = 8  # Maximum number of clusters to test
    iterations = 10  # Number of iterations for K-Means

    # Store inertia values for each k
    inertia_values = []

    # Create a column for each k
    cols = st.columns(max_k)
    for k in range(1, max_k + 1):
        with cols[k - 1]:
            st.write(f"K = {k}")
            animations, inertia = k_means_clustering(data, k, iterations)
            inertia_values.append(inertia)
            st.plotly_chart(create_animation(animations), use_container_width=True)

    # Plot inertia vs k
    st.write("### Inertia vs Number of Clusters (K)")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(1, max_k + 1)), y=inertia_values, mode='lines+markers', name='Inertia'
    ))
    fig.update_layout(
        xaxis_title="Number of Clusters (K)",
        yaxis_title="Inertia",
        title="Inertia as a Function of K"
    )
    st.plotly_chart(fig, use_container_width=True)
