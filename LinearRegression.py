import numpy as np

# Linear regression implementation
class LinearRegression:
    def __init__(self, learning_rate=0.01, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.m = 0  # slope
        self.b = 0  # intercept
    
    def fit(self, X, y):
        n = len(X)
        
        # Gradient descent
        for _ in range(self.epochs):
            y_pred = self.m * X + self.b
            # Calculate gradients
            dm = (-2 / n) * np.sum(X * (y - y_pred))
            db = (-2 / n) * np.sum(y - y_pred)
            
            # Update parameters
            self.m -= self.learning_rate * dm
            self.b -= self.learning_rate * db
    
    def predict(self, X):
        return self.m * X + self.b

# Example usage
if __name__ == "__main__":
    # Sample dataset
    X = np.array([1, 2, 3, 4, 5], dtype=float)
    y = np.array([2, 4, 5, 4, 5], dtype=float)
    
    # Model initialization
    model = LinearRegression(learning_rate=0.01, epochs=2000)
    
    # Training the model
    model.fit(X, y)
    
    # Predictions
    predictions = model.predict(X)
    
    print(f"Slope (m): {model.m}")
    print(f"Intercept (b): {model.b}")
    print(f"Predictions: {predictions}")
