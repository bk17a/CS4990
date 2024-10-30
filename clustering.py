import random
import math
from collections import deque
import numpy as np


def dist(center, instance, columns):
    return math.sqrt(
        sum((center[i] - instance[columns[i]]) ** 2 for i in range(len(columns)))
    )


# DO NOT CHANGE THE FOLLOWING LINE
def lloyds(data, k, columns, centers=None, n=None, eps=None):
    # DO NOT CHANGE THE PRECEDING LINE
    # This function has to return a list of k cluster centers (lists of floats of the same length as columns)

    # Step 1: Randomly choose centers if not initalized already
    if centers is None:
        centers = [random.choice(data) for _ in range(k)]
    else:
        centers = [list(center) for center in centers]

    it = 0
    # Step 2: Repeat
    while True:
        # Step 3: (Re)assign each data point to the nearest cluster
        clusters = [[] for _ in range(k)]
        # Loop over each data point in the data set
        for instance in data:
            # Calculate distance from current data point to each cluster center
            distances = [dist(center, instance, columns) for center in centers]

            # Find the index of the nearest cluster center
            closest_center = distances.index(min(distances))

            # Assign the instance to the closest cluster's list in clusters
            clusters[closest_center].append(instance)

        # Step 4: Calculate new centers and track total movement
        new_centers = []
        total_movement = 0
        for i, cluster in enumerate(clusters):
            if not cluster:  # Handle empty cluster
                new_centers.append(centers[i])  # Keep old center if cluster is empty
            else:
                # Calculate mean for each column in cluster
                new_center = [
                    sum(point[columns[j]] for point in cluster) / len(cluster)
                    for j in range(len(columns))
                ]
                total_movement += dist(centers[i], new_center, range(len(columns)))
                new_centers.append(new_center)

        # Update centers
        centers = new_centers
        it += 1

        # Stopping conditions
        if (n is not None and it >= n) or (eps is not None and total_movement < eps):
            break

    return centers


# DO NOT CHANGE THE FOLLOWING LINE
def dbscan(data, columns, eps, min_samples):
    # DO NOT CHANGE THE PRECEDING LINE
    # This function has to return a list of cluster centers (lists of floats of the same length as columns)
    data = np.array([d[columns] for d in data])
    tags = [-1] * len(data)
    cid = 0

    def get_neighbors(p):
        nbs = []
        for i, pt in enumerate(data):
            if np.linalg.norm(data[p] - pt) <= eps:
                nbs.append(i)
        return nbs

    def build_cluster(idx, nbs):
        tags[idx] = cid
        q = deque(nbs)

        while q:
            n = q.popleft()
            if tags[n] == -1:
                tags[n] = cid
                new_nbs = get_neighbors(n)
                if len(new_nbs) >= min_samples:
                    q.extend(new_nbs)
            elif tags[n] == 0:
                tags[n] = cid

    for i in range(len(data)):
        if tags[i] != -1:
            continue
        nbs = get_neighbors(i)
        if len(nbs) >= min_samples:
            cid += 1
            build_cluster(i, nbs)
        else:
            tags[i] = 0

    clusters = [[] for _ in range(cid)]
    noise = []
    for idx, label in enumerate(tags):
        if label > 0:
            clusters[label - 1].append(data[idx])
        else:
            noise.append(data[idx])

    return clusters, noise


# DO NOT CHANGE THE FOLLOWING LINE
def kmedoids(data, k, distance, centers=None, n=None, eps=None):
    # DO NOT CHANGE THE PRECEDING LINE
    # This function has to return a list of k cluster centroids (data instances!)
    if centers is None:
        centers = random.sample(data, k)
    else:
        centers = centers[:]

    it = 0
    while True:
        clusters = [[] for _ in range(k)]
        for instance in data:
            distances = [distance(center, instance) for center in centers]
            closest_center = distances.index(min(distances))
            clusters[closest_center].append(instance)

        total_movement = 0
        for i in range(k):
            if not clusters[i]:  # Empty cluster
                continue
            medoid, min_cost = None, float("inf")
            for candidate in clusters[i]:
                cost = sum(distance(candidate, other) for other in clusters[i])
                if cost < min_cost:
                    medoid, min_cost = candidate, cost
            total_movement += distance(centers[i], medoid)
            centers[i] = medoid

        it += 1
        if (n is not None and it >= n) or (eps is not None and total_movement < eps):
            break

    return centers


# Helper distance function for k-medoids
def manhattan_dist(instance1, instance2):
    return sum(abs(a - b) for a, b in zip(instance1, instance2))
