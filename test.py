from collections import deque
import numpy as np
import matplotlib.pyplot as plt


# DBSCAN Implementation
def mydbscan(data, cols, eps, min_samples):
    # Extract relevant columns for clustering
    data = np.array([d[cols] for d in data])
    tags = [-1] * len(data)  # Tag each point as unclassified
    cid = 0  # Cluster ID

    def get_neighbors(p):
        """Find neighbors within 'eps' distance."""
        nbs = []
        for i, pt in enumerate(data):
            if np.linalg.norm(data[p] - pt) <= eps:
                nbs.append(i)
        return nbs

    def build_cluster(idx, nbs):
        """Expand cluster starting from point idx."""
        tags[idx] = cid
        q = deque(nbs)

        while q:
            n = q.popleft()
            if tags[n] == -1:  # Not yet classified
                tags[n] = cid
                new_nbs = get_neighbors(n)

                if len(new_nbs) >= min_samples:
                    q.extend(new_nbs)

            elif tags[n] == 0:  # Previously labeled noise, reassign to cluster
                tags[n] = cid

    # DBSCAN algorithm main loop
    for i in range(len(data)):
        if tags[i] != -1:
            continue

        nbs = get_neighbors(i)
        if len(nbs) >= min_samples:
            cid += 1
            build_cluster(i, nbs)
        else:
            tags[i] = 0  # Label as noise

    # Separate data points into clusters and noise
    clusts = [[] for _ in range(cid)]
    noise = []

    for idx, label in enumerate(tags):
        if label > 0:
            clusts[label - 1].append(data[idx])
        else:
            noise.append(data[idx])

    return clusts, noise


# Test Data Setup
data = np.array(
    [
        [1, 1],
        [1, 2],
        [2, 1],
        [2, 2],
        [1.5, 1.5],  # Cluster 1
        [5, 5],
        [5, 6],
        [6, 5],
        [6, 6],
        [5.5, 5.5],  # Cluster 2
        [9, 9],
        [9, 10],
        [10, 9],
        [10, 10],
        [9.5, 9.5],  # Cluster 3
        [0, 0],
        [10, 0],
        [5, 1],
        [1, 5],
        [7, 7],  # Noise points
    ]
)

# DBSCAN Parameters
cols = [0, 1]  # Using both x and y columns for clustering
eps = 1.5  # epsius for density search
min_samples = 3  # Minimum number of points required to form a dense region

# Run DBSCAN
clusters, noise = mydbscan(data, cols, eps, min_samples)

# Visualization
for i, cluster in enumerate(clusters, 1):
    cluster = np.array(cluster)
    plt.scatter(cluster[:, 0], cluster[:, 1], label=f"Cluster {i}")

noise = np.array(noise)
plt.scatter(noise[:, 0], noise[:, 1], color="k", label="Noise")

plt.legend()
plt.xlabel("X")
plt.ylabel("Y")
plt.title("DBSCAN Clustering Results")
plt.show()
