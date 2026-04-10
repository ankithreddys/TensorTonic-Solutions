import numpy as np
def k_means_assignment(points, centroids):
    """
    Assign each point to the nearest centroid.
    """
    # Write code here
    points = np.asarray(points)
    centroids = np.asarray(centroids)
    def euclidian_distance(p, c):
        return np.sqrt(sum((p-c)**2))

    assignments = []
    for p in points:
        dist = np.inf
        centroid_index = -1
        for i in range(len(centroids)):
            e_d = euclidian_distance(p, centroids[i])
            if e_d < dist:
                dist = e_d
                centroid_index = i
        assignments.append(centroid_index)

    return assignments
            
            
            
            