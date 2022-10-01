import numpy as np


def find_closest_centroids(X, centroids):
    """
    Computes the centroid memberships for every example.

    Args:
        X (ndarray): (m, n) Input values
        centroids (ndarray): k centroids

    Returns:
        x_centroids_idx (array_like): (m,) closest centroids

    """
    x_centroids_idx = np.zeros(X.shape[0], dtype=int)
    for x_i, x in enumerate(X):
        smallest_distance = float('inf')
        closest_centroid = None
        for k, centroid in enumerate(centroids):
            norm = np.linalg.norm(x - centroid) ** 2
            if norm < smallest_distance:
                smallest_distance = norm
                closest_centroid = k
        x_centroids_idx[x_i] = closest_centroid
    return x_centroids_idx


def update_centroids(X, idx, K):
    """
    Returns the new centroids by computing the means of the
    data points assigned to each centroid.

    Args:
        X (ndarray):   (m, n) Data points
        idx (ndarray): (m,) Array containing index of the closest centroid for each
                       example in X. Concretely, idx[i] contains the index of
                       the centroid closest to example i
        K (int):       number of centroids

    Returns:
        centroids (ndarray): (K, n) New centroids computed
    """
    centroids = np.zeros((K, X.shape[1]))
    for c_i in range(K):
        x_sum = np.zeros(X.shape[1])
        x_cnt = 0
        for x_i, x in enumerate(X):
            if idx[x_i] == c_i:
                x_cnt += 1
                x_sum += x
        centroids[c_i] = x_sum / x_cnt
    return centroids


def run_k_means(X, k, max_iters=10):
    """
    Runs the K-Means algorithm on data matrix X, where each row of X
    is a single example. k is a number of desired centroids (clusters).
    """
    # Randomly reorder the indices of examples
    randidx = np.random.permutation(X.shape[0])
    # Take the first K examples as centroids
    centroids = X[randidx[:k]]

    # Run K-Means
    max_iters = max(1, max_iters)
    for i in range(max_iters):
        # For each example in X, assign it to the closest centroid
        x_centroids_idx = find_closest_centroids(X, centroids)
        # Given the memberships, compute new centroids
        centroids = update_centroids(X, x_centroids_idx, k)
    return centroids, x_centroids_idx
