import streamlit as st
import numpy as np
import plotly.graph_objects as go

st.title("Interaktiv K-means clustering")
st.write("K-means clustering er en algoritme som bruges til at gruppere unlabeled data. Dette kan blandt andet bruges til at lave et anbefalingsystem til en streamingplatform eller segmentering af kunder. Algoritmen virker ud fra følgende princip:")
st.write("""
1.    Vælg ***k*** tilfældige punkter fra datasættet som centrum til ***k*** clusters
2.    Tildel hvert punkt til det nærmeste centrum
3.    Find centroiden for hver cluster ved at tage det gennemsnitlige x- og y -værdi, inden for hvert cluster, og lad dette være det nye centrum
4.    Gentag punkt 2-3 indtil centroiderne ikke ændre sig, eller antallet af iterationer er nået.
""")
st.write("Nedenunder ses den interaktive K-means clustering algoritme. For at starte skal der laves nogle punkter. Dette kan gøres ved at vælge værdier i menuen til venstre og generer punkterne. Der er lavet en animation til K=1 til K=8. For at se udviklingen gennem animationen kan du trykke på \"Afspil\" under hver graf")

if "clicked_points" not in st.session_state:
    st.session_state.clicked_points = []

st.sidebar.header("Generer clusters")

# User-defined cluster parameters
x_center = st.sidebar.number_input("X-værdi for clustercenter", min_value=0, max_value=100, step=1)
y_center = st.sidebar.number_input("Y-værdi for clustercenter", min_value=0, max_value=100, step=1)
num_points = st.sidebar.number_input("Antal punkter i cluster", min_value=1, max_value=100, value=20, step=1)
spread = st.sidebar.number_input("Spredning(lavere værdier giver tættere pakket clusters)", min_value=1, max_value=30, value=2, step=1)

# Button to generate a single cluster
if st.sidebar.button("Generer Cluster"):
    points = np.random.normal(loc=[x_center, y_center], scale=spread, size=(num_points, 2))
    st.session_state.clicked_points.extend(points.tolist())

# Random cluster generation function
def generate_random_clusters(n_clusters, points_per_cluster, spread):
    for _ in range(n_clusters):
        x_rand = np.random.uniform(0, 100)
        y_rand = np.random.uniform(0, 100)
        points = np.random.normal(loc=[x_rand, y_rand], scale=spread, size=(points_per_cluster, 2))
        st.session_state.clicked_points.extend(points.tolist())

# User input for random cluster generation
st.sidebar.header("Tilfældig generering af clusters")
n_clusters = st.sidebar.number_input("Antal clusters", min_value=1, max_value=10, value=3, step=1)
rand_points_per_cluster = st.sidebar.number_input("Punkter per cluster", min_value=1, max_value=100, value=20, step=1)
rand_spread = st.sidebar.number_input("Spredning af tilfældige clusters", min_value=1, max_value=30, value=2, step=1)

if st.sidebar.button("Generer Tilfældige Clusters"):
    generate_random_clusters(n_clusters, rand_points_per_cluster, rand_spread)

def k_means_clustering(data, k, iterations=10):
    centroids = data[np.random.choice(len(data), k, replace=False)]
    animations = []
    for _ in range(iterations):
        distances = np.sqrt(((data - centroids[:, np.newaxis])**2).sum(axis=2))
        labels = np.argmin(distances, axis=0)
        animation_step = {'centroids': centroids.copy(), 'labels': labels}
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
    fig.add_trace(go.Scatter(x=clicked_x, y=clicked_y, mode='markers', name='Punkter', marker=dict(color='gray', size=5)))
    fig.add_trace(go.Scatter(x=[], y=[], mode='markers', name='Centroid', marker=dict(color='black', symbol='x', size=10)))
    frames = []
    for iteration, animation_step in enumerate(animations):
        centroids = animation_step['centroids']
        labels = animation_step['labels']
        colors = [f'hsl({(i * 360 / len(centroids)) % 360}, 100%, 50%)' for i in range(len(centroids))]
        point_colors = [colors[label] for label in labels]
        frame = go.Frame(data=[go.Scatter(x=clicked_x, y=clicked_y, marker=dict(color=point_colors)),
                               go.Scatter(x=centroids[:, 0], y=centroids[:, 1])], name=f'Frame {iteration}')
        frames.append(frame)
    fig.frames = frames
    fig.update_layout(xaxis_title="X-akse", yaxis_title="Y-akse", xaxis=dict(range=[0, 100], autorange=False), yaxis=dict(range=[0, 100], autorange=False),
                      updatemenus=[{'buttons': [{'args': [None, {'frame': {'duration': 1000, 'redraw': True}, 'fromcurrent': True}],
                                                 'label': 'Afspil', 'method': 'animate'},
                                                {'args': [[None], {'frame': {'duration': 0, 'redraw': True}, 'mode': 'immediate', 'transition': {'duration': 0}}],
                                                 'label': 'Pause', 'method': 'animate'}],
                                    'direction': 'left', 'pad': {'r': 10, 't': 87}, 'showactive': False, 'type': 'buttons', 'x': 0.1, 'xanchor': 'right', 'y': 0, 'yanchor': 'top'}])
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
    st.write("### Albue-metoden")
    st.write("For at finde den optimale værdi for K, bruges albue-metoden som sammenligner afstandene fra punkter til centroid, også kaldt SSE(Sum of squared errors), for hver værdi af K. Der hvor man ser \"knækket på albuen\" er den værdi for K som vil være optimal. Det er altså ikke den laveste værdi som nødvendigvis er den mest passende.")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(range(1, max_k + 1)), y=inertia_values, mode='lines+markers', name='SSE'))
    fig.update_layout(xaxis_title="Antal Clusters (K)", yaxis_title="SSE", title="SSE for hver værdi af K")
    st.plotly_chart(fig, use_container_width=True)
