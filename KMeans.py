import numpy as np

class KMeans:
    def __init__(self, n_clusters=3, max_iter=300, tol=1e-4):
        """
        Initialize the KMeans class.
        :param n_clusters: Number of clusters
        :param max_iter: Maximum number of iterations
        :param tol: Tolerance for convergence
        """
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.centroids = None

    def fit(self, data):
        """
        Fit the KMeans model to the data.
        :param data: Array-like dataset (rows = samples, columns = features)
        """
        # Randomly initialize centroids
        np.random.seed(42)
        self.centroids = data[np.random.choice(data.shape[0], self.n_clusters, replace=False)]

        for i in range(self.max_iter):
            # Assign clusters based on closest centroid
            self.labels = self._assign_clusters(data)
            
            # Update centroids
            new_centroids = self._compute_centroids(data)
            
            # Check for convergence
            if np.all(np.linalg.norm(new_centroids - self.centroids, axis=1) < self.tol):
                break
            self.centroids = new_centroids

    def predict(self, data):
        """
        Predict the closest cluster each sample in data belongs to.
        :param data: Array-like dataset
        :return: Array of cluster labels
        """
        return self._assign_clusters(data)

    def _assign_clusters(self, data):
        """
        Assign data points to the nearest cluster centroid.
        :param data: Array-like dataset
        :return: Array of cluster indices
        """
        distances = np.linalg.norm(data[:, np.newaxis] - self.centroids, axis=2)
        return np.argmin(distances, axis=1)

    def _compute_centroids(self, data):
        """
        Compute the centroids of the clusters.
        :param data: Array-like dataset
        :return: New centroids
        """
        centroids = np.array([data[self.labels == i].mean(axis=0) for i in range(self.n_clusters)])
        return centroids


# Example Usage
if __name__ == "__main__":
    # Generate sample data
    from sklearn.datasets import make_blobs

    X, _ = make_blobs(n_samples=300, centers=3, cluster_std=1.0, random_state=42)

    # Initialize and fit KMeans
    kmeans = KMeans(n_clusters=3)
    kmeans.fit(X)

    # Predict clusters
    labels = kmeans.predict(X)

    # Visualize the results
    import matplotlib.pyplot as plt

    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
    plt.scatter(kmeans.centroids[:, 0], kmeans.centroids[:, 1], s=200, c='red', marker='X')  # Centroids
    plt.title("K-Means Clustering")
    plt.show()
