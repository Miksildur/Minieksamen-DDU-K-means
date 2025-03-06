import streamlit as st
import numpy as np
import plotly.graph_objects as go

st.title("Interaktiv K-means clustering")
st.write("Indtast et eller flere koordinatsæt til klyngerne, og se udviklingen af algoritmen ved at trykke på afspil")

if "clicked_points" not in st.session_state:
    st.session_state.clicked_points = []

st.sidebar.header("Generér clusters")
x_center = st.sidebar.number_input("X-værdi for clustercenter", min_value=0, max_value=100, step=1)
y_center = st.sidebar.number_input("Y-værdi for clustercenter", min_value=0, max_value=100, step=1)

num_points = st.sidebar.number_input("Antal punkter i cluster", min_value=1, max_value=100, value=20, step=1)
spread = st.sidebar.number_input("Spredning(lavere værdier giver tættere pakket clusters)", min_value=1, max_value=30, value=5, step=1)

if st.sidebar.button("Generér Cluster"):
    points = np.random.normal(loc=[x_center, y_center], scale=spread, size=(num_points, 2))
    st.session_state.clicked_points.extend(points.tolist())

def k_means_clustering(data, k, iterations=10):
    centroids = data[np.random.choice(len(data), k, replace=False)]
    animations = []

    for _ in range(iterations):
        distances = np.sqrt(((data - centroids[:, np.newaxis])**2).sum(axis=2))
        labels = np.argmin(distances, axis=0)

        animation_step = {
            'centroids': centroids.copy(),
            'labels': labels
        }
        animations.append(animation_step)

        new_centroids = np.array([data[labels == i].mean(axis=0) if np.sum(labels == i) > 0 else centroids[i] for i in range(k)])
        centroids = new_centroids

    inertia = np.sum(np.min(distances, axis=0)**2)
    return animations, inertia

def create_animation(animations):
    data = np.array(st.session_state.clicked_points)
    clicked_x = data[:, 0]
    clicked_y = data[:, 1]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=clicked_x, y=clicked_y, mode='markers', name='Punkter',
        marker=dict(color='gray', size=5)
    ))

    fig.add_trace(go.Scatter(
        x=[], y=[], mode='markers', name='Centroid',
        marker=dict(color='black', symbol='x', size=10)
    ))

    frames = []
    for iteration, animation_step in enumerate(animations):
        centroids = animation_step['centroids']
        labels = animation_step['labels']

        colors = [f'hsl({(i * 360 / len(centroids)) % 360}, 100%, 50%)' for i in range(len(centroids))]
        point_colors = [colors[label] for label in labels]

        frame = go.Frame(
            data=[
                go.Scatter(x=clicked_x, y=clicked_y, marker=dict(color=point_colors)),
                go.Scatter(x=centroids[:, 0], y=centroids[:, 1])
            ],
            name=f'Frame {iteration}'
        )
        frames.append(frame)

    fig.frames = frames

    fig.update_layout(
        xaxis_title="X-akse",
        yaxis_title="Y-akse",
        xaxis=dict(range=[0, 100], autorange=False),
        yaxis=dict(range=[0, 100], autorange=False),
        updatemenus=[{
            'buttons': [
                {
                    'args': [None, {'frame': {'duration': 1000, 'redraw': True}, 'fromcurrent': True}],
                    'label': 'Afspil',
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

if len(st.session_state.clicked_points) > 1:
    data = np.array(st.session_state.clicked_points)
    max_k = 8
    iterations = 10

    inertia_values = []

    for k in range(1, max_k + 1):
        st.write(f"### K-means clustering med K = {k}")
        animations, inertia = k_means_clustering(data, k, iterations)
        inertia_values.append(inertia)
        st.plotly_chart(create_animation(animations), use_container_width=True)

    st.write("### Inerti vs Antal Klynger (K)")
    st.write("Der hvor man ser \"albuen\" knække, er den værdi for K, som nok vil være mest passende")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(1, max_k + 1)), y=inertia_values, mode='lines+markers', name='Inerti'
    ))
    fig.update_layout(
        xaxis_title="Antal Klynger (K)",
        yaxis_title="Inerti",
        title="Inerti som en Funktion af K"
    )
    st.plotly_chart(fig, use_container_width=True)
