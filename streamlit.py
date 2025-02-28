import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

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
    centroids = data[np.random.choice(len(data), k, replace=False)]  # Random initial centroids
    animations = []

    for iteration in range(iterations):
        classes = {tuple(c): [] for c in centroids}
        for point in data:
            closest_centroid = min(centroids, key=lambda c: np.linalg.norm(point - c))
            classes[tuple(closest_centroid)].append(point)

        # Calculate new centroids
        centroids = [np.mean(points, axis=0) if points else centroid for centroid, points in classes.items()]
        
        # Store the current step for animation
        animation_step = {
            'centroids': centroids,
            'classes': classes
        }
        animations.append(animation_step)

    return animations

# --- Create Plotly Figure ---
def create_animation(animations):
    fig = go.Figure()

    # Initial points
    clicked_x, clicked_y = zip(*st.session_state.clicked_points)
    fig.add_trace(go.Scatter(
        x=clicked_x, y=clicked_y, mode='markers', name='Points',
        marker=dict(color='gray', size=5)
    ))

    # Add empty frames for animation
    frames = []
    
    for iteration, animation_step in enumerate(animations):
        centroids = np.array(animation_step['centroids'])
        classes = animation_step['classes']
        
        # Add clusters for this frame
        cluster_traces = []
        for i, (centroid, points) in enumerate(classes.items()):
            points = np.array(points)
            cluster_traces.append(go.Scatter(
                x=points[:, 0], y=points[:, 1], mode='markers',
                name=f'Cluster {i+1}', marker=dict(color=f'rgb({i*50},{255-i*50},150)', size=6)
            ))

        # Add centroids for this frame
        centroid_trace = go.Scatter(
            x=centroids[:, 0], y=centroids[:, 1], mode='markers', name='Centroids',
            marker=dict(color='black', symbol='x', size=10)
        )
        cluster_traces.append(centroid_trace)

        # Create frame with cluster and centroid positions
        frames.append(go.Frame(
            data=cluster_traces,
            name=f'Frame {iteration}'
        ))

    fig.frames = frames

    # Update layout for animation
    fig.update_layout(
        title="K-Means Clustering Animation",
        xaxis_title="X-axis",
        yaxis_title="Y-axis",
        xaxis=dict(range=[0, 100]),  # Manually setting the axis range
        yaxis=dict(range=[0, 100]),
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

# Perform K-Means clustering with animation steps
if len(st.session_state.clicked_points) > 1:
    data = np.array(st.session_state.clicked_points)
    k = 4  # Number of clusters
    animations = k_means_clustering(data, k, iterations=10)
    
    # Create the plot animation
    st.plotly_chart(create_animation(animations), use_container_width=True)
