import numpy as np
import streamlit as st
import plotly.graph_objects as go

# --- Helper functions for K-means ---
def generate_random_dataset(k=1, n=100, spread=3.0, seed=42):
    """Generates a dataset with `k` random clusters."""
    np.random.seed(seed)
    clusters = []
    centers = np.random.uniform(0, 50, size=(k, 2))  # Cluster centers in range [0, 50]
    
    for center in centers:
        points = center + spread * np.random.randn(n, 2)
        clusters.append(points)

    return np.vstack(clusters), centers

def find_closest_centroid(point, centroids):
    closest_centroid = None
    closest_distance = None
    for centroid in centroids:
        distance = np.linalg.norm(point - centroid)
        if closest_distance is None or distance < closest_distance:
            closest_distance = distance
            closest_centroid = centroid
    return closest_centroid

def update_centroids(centroids, classes):
    updated_centroids = []
    for centroid in centroids:
        if len(classes[tuple(centroid)]) > 0:
            x_mean = np.mean([point[0] for point in classes[tuple(centroid)]])
            y_mean = np.mean([point[1] for point in classes[tuple(centroid)]])
            updated_centroids.append((x_mean, y_mean))
        else:
            updated_centroids.append(centroid)
    return updated_centroids

def k_means_clustering(data, k, iterations):
    centroids = []
    all_classes = []
    all_centroids = []
    
    # Initialize centroids randomly
    centroids.append(data[np.random.randint(0, len(data))])  # First centroid randomly
    for _ in range(1, k):
        distances = np.array([min([np.linalg.norm(point - c) for c in centroids]) for point in data])
        probabilities = distances / distances.sum()
        centroids.append(data[np.random.choice(len(data), p=probabilities)])

    for _ in range(iterations):
        # Reset classes dictionary
        classes = {tuple(centroid): [] for centroid in centroids}
        
        # Assign points to the closest centroid
        for point in data:
            closest_centroid = find_closest_centroid(point, centroids)
            classes[tuple(closest_centroid)].append(point)
        
        all_classes.append(classes.copy())  # Store classes for animation
        all_centroids.append(centroids.copy())  # Store centroids for animation

        # Update centroids
        centroids = update_centroids(centroids, classes)

    return all_centroids, all_classes

# --- Streamlit setup ---
st.title("K-Means Clustering Animation with Plotly")

# Input for the number of clusters
k = st.sidebar.number_input("Number of Clusters", min_value=1, max_value=20, value=8, step=1)
n = st.sidebar.number_input("Number of Points per Cluster", min_value=10, max_value=200, value=50, step=10)
iterations = st.sidebar.number_input("Number of Iterations", min_value=1, max_value=50, value=10, step=1)
spread = st.sidebar.number_input("Spread of Clusters", min_value=0, max_value=10, value=3, step=1)

# Generate random dataset
data, initial_centers = generate_random_dataset(k=k, n=n, spread=spread)

# Perform K-Means clustering
all_centroids, all_classes = k_means_clustering(data, k, iterations)

# --- Plotly Animation ---
def create_animation(animations):
    fig = go.Figure()
    colors = ['red', 'blue', 'green', 'cyan', 'magenta', 'yellow', 'orange', 'pink', 'purple', 'brown']  # Adjust colors for more clusters
    frames = []

    # Add points as a base trace
    clicked_x, clicked_y = zip(*data)
    fig.add_trace(go.Scatter(
        x=clicked_x, y=clicked_y, mode='markers', name='Data Points',
        marker=dict(color='gray', size=5)
    ))

    # Generate frames for animation
    for iteration, animation_step in enumerate(animations):
        centroids = np.array(animation_step['centroids'])
        classes = animation_step['classes']
        
        cluster_traces = []
        # Add points for each cluster
        for i, (centroid, points) in enumerate(classes.items()):
            points = np.array(points)
            cluster_traces.append(go.Scatter(
                x=points[:, 0], y=points[:, 1], mode='markers',
                name=f'Cluster {i+1}', marker=dict(color=colors[i % len(colors)], size=6)
            ))

        # Add centroids for this frame
        centroid_trace = go.Scatter(
            x=centroids[:, 0], y=centroids[:, 1], mode='markers', name='Centroids',
            marker=dict(color='black', symbol='x', size=10)
        )
        cluster_traces.append(centroid_trace)

        # Create frame with clusters and centroids
        frames.append(go.Frame(data=cluster_traces, name=f'Frame {iteration}'))

    fig.frames = frames

    fig.update_layout(
        title="K-Means Clustering Animation",
        xaxis_title="X-axis",
        yaxis_title="Y-axis",
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

# Create the plot animation with frames
animations = [{'centroids': all_centroids[i], 'classes': all_classes[i]} for i in range(iterations)]
fig = create_animation(animations)

# Show the plotly animation in Streamlit
st.plotly_chart(fig, use_container_width=True)
