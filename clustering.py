import random
import math


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
    pass


# DO NOT CHANGE THE FOLLOWING LINE
def kmedoids(data, k, distance, centers=None, n=None, eps=None):
    # DO NOT CHANGE THE PRECEDING LINE
    # This function has to return a list of k cluster centroids (data instances!)
    pass
