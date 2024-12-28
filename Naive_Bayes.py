import numpy as np

class NaiveBayes:
    def __init__(self):
        self.classes = None
        self.mean = {}
        self.var = {}
        self.priors = {}

    def fit(self, X, y):
        """
        Train the Naive Bayes classifier.
        X: 2D numpy array (features)
        y: 1D numpy array (labels)
        """
        self.classes = np.unique(y)
        for cls in self.classes:
            X_c = X[y == cls]
            self.mean[cls] = X_c.mean(axis=0)
            self.var[cls] = X_c.var(axis=0)
            self.priors[cls] = X_c.shape[0] / X.shape[0]

    def _calculate_likelihood(self, cls, x):
        """
        Calculate the Gaussian likelihood.
        """
        mean = self.mean[cls]
        var = self.var[cls]
        numerator = np.exp(-((x - mean) ** 2) / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)
        return numerator / denominator

    def _calculate_posterior(self, x):
        """
        Calculate the posterior probability for each class.
        """
        posteriors = []
        for cls in self.classes:
            prior = np.log(self.priors[cls])
            likelihood = np.sum(np.log(self._calculate_likelihood(cls, x)))
            posterior = prior + likelihood
            posteriors.append(posterior)
        return self.classes[np.argmax(posteriors)]

    def predict(self, X):
        """
        Predict the class for each sample in X.
        """
        return np.array([self._calculate_posterior(x) for x in X])


# Example usage:
# Sample data
X = np.array([[1.0, 2.0], [1.5, 1.8], [5.0, 8.0], [8.0, 8.0], [1.0, 0.6], [9.0, 11.0]])
y = np.array([0, 0, 1, 1, 0, 1])

# Train the model
nb = NaiveBayes()
nb.fit(X, y)

# Test data
X_test = np.array([[2.0, 3.0], [6.0, 9.0]])
predictions = nb.predict(X_test)

print("Predictions:", predictions)
