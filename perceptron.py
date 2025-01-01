import numpy as np

class Perceptron:
    def __init__(self, input_size, learning_rate=0.01, epochs=1000):
        self.weights = np.zeros(input_size + 1)  # Including bias
        self.learning_rate = learning_rate
        self.epochs = epochs

    def activation_function(self, x):
        return 1 if x >= 0 else 0

    def predict(self, x):
        weighted_sum = np.dot(x, self.weights[1:]) + self.weights[0]  # w.x + bias
        return self.activation_function(weighted_sum)

    def fit(self, X, y):
        for _ in range(self.epochs):
            for inputs, label in zip(X, y):
                prediction = self.predict(inputs)
                error = label - prediction
                self.weights[1:] += self.learning_rate * error * inputs
                self.weights[0] += self.learning_rate * error  # Update bias

# Example Dataset: OR Gate
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 1, 1, 1])  # Output for OR gate

# Initialize and train the perceptron
perceptron = Perceptron(input_size=2, learning_rate=0.1, epochs=10)
perceptron.fit(X, y)

# Test the perceptron
for inputs in X:
    print(f"Input: {inputs}, Predicted Output: {perceptron.predict(inputs)}")
