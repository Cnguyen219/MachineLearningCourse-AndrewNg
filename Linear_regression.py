import numpy as np

class LinearRegression:
    def __init__(self, learning_rate=0.01, iterations=1000):
        # learning_rate (alpha) controls how big of a step we take during gradient descent
        self.learning_rate = learning_rate
        
        # Number of times we update w and b
        self.iterations = iterations
        
        # Initialize model parameters (slope and intercept)
        self.w = 0  # weight (slope)
        self.b = 0  # bias (intercept)

    def compute_cost(self, X, y):
        """
        Computes Mean Squared Error cost:
        J(w,b) = (1 / 2m) * sum((prediction - y)^2)
        """
        m = len(X)  # number of training examples
        
        # Compute predictions using current w and b
        predictions = self.w * X + self.b
        
        # Compute squared errors
        errors = predictions - y
        
        # Compute cost
        cost = (1 / (2 * m)) * np.sum(errors ** 2)
        
        return cost

    def fit(self, X, y):
        """
        Performs Gradient Descent to learn w and b
        """
        m = len(X)  # number of training examples

        for _ in range(self.iterations):
            
            # Step 1: Compute predictions
            predictions = self.w * X + self.b
            
            # Step 2: Compute gradients (partial derivatives)
            # derivative of cost w.r.t w
            dw = (1 / m) * np.sum((predictions - y) * X)
            
            # derivative of cost w.r.t b
            db = (1 / m) * np.sum(predictions - y)
            
            # Step 3: Update parameters using gradient descent rule
            self.w -= self.learning_rate * dw
            self.b -= self.learning_rate * db

    def predict(self, X):
        """
        Uses learned w and b to predict new values
        """
        return self.w * X + self.b
