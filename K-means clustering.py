import numpy as np
import matplotlib.pyplot as plt


def generate_random_dataset(k, n, spread, seed=42):
    """
    Generates a dataset with `k` random clusters.
    
    Parameters:
        k (int): Number of clusters
        n (int): Number of points per cluster
        spread (float): Spread of points around the cluster center
        seed (int): Random seed for reproducibility

    Returns:
        np.ndarray: A dataset of points
    """
    #np.random.seed(seed)  # For reproducibility
    clusters = []
    
    # Generate `k` random cluster centers within a specified range
    centers = np.random.uniform(0, 100, size=(k, 2))  # Centers in range [-10, 10]
    
    for center in centers:
        # Generate `n` points around the center using a normal distribution
        points = center + spread * np.random.randn(n, 2)
        clusters.append(points)

    return np.vstack(clusters)  # Stack all clusters into a single dataset

# --- Generate Dataset ---
data = generate_random_dataset(k=5, n=300, spread=10)

def find_closest_centroid(point, centroids):    
    closest_centroid = None
    closest_distance = None
    for centroid in centroids:
        distance = np.sqrt((point[0] - centroid[0]) ** 2 + (point[1] - centroid[1]) ** 2)
        if closest_distance is None or distance < closest_distance:
            closest_distance = distance
            closest_centroid = centroid
    return closest_centroid

def update_centroids(centroids, classes):
    updated_centroids = []
    for centroid in centroids:
        x_sum = 0
        y_sum = 0
        for point in classes[centroid]:
            x_sum += point[0]
            y_sum += point[1]
        x_mean = x_sum / len(classes[centroid])
        y_mean = y_sum / len(classes[centroid])
        updated_centroids.append((float(x_mean), float(y_mean)))
    return updated_centroids

def k_means_clustering(data, k, iterations):
    centroids = []
    classes = {}
    
    # Initialize centroids as tuples
    for i in range(k):
        centroid = tuple(data[np.random.randint(0, len(data))])  # Convert to tuple
        centroids.append(centroid)
        classes[centroid] = []  # Use tuple as dictionary key
    for i in range(iterations):
        # Reset classes dictionary
        classes = {centroid: [] for centroid in centroids}
        for point in data:
            closest_centroid = find_closest_centroid(point, centroids)
            classes[closest_centroid].append((int(point[0]), int(point[1])))
        centroids = update_centroids(centroids, classes)
    return centroids, classes


k = 4
iterations = 10
centroids, classes = k_means_clustering(data, k, iterations)

plt.figure(figsize=(8, 6))

# Define color map
colors = ['r', 'b', 'g', 'c', 'm', 'y']  # Supports up to 6 clusters

# Plot each cluster with a unique color
for i, (centroid, points) in enumerate(classes.items()):
    points = np.array(points)
    plt.scatter(points[:, 0], points[:, 1], color=colors[i % len(colors)], label=f'Cluster {i+1}')

# Plot centroids
centroids = np.array(centroids)
plt.scatter(centroids[:, 0], centroids[:, 1], color='black', marker='X', s=200, label='Centroids')

# Labels and legend
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.title("K-Means Clustering")
plt.legend()
plt.grid(True)
plt.show()
