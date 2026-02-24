import numpy as np

class LinearRegression:
  def __init__(self, learning_rate=0.01, iterations=1000):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.w = 0
        self.b = 0
  def compute_cost(self, X, y):
        m = len(X)
        predictions = self.w * X + self.b
        cost = (1 / (2 * m)) * np.sum((predictions - y) ** 2)
        return cost

 def fit(self, X, y):
        m = len(X)

        for _ in range(self.iterations):
            predictions = self.w * X + self.b

            dw = (1 / m) * np.sum((predictions - y) * X)
            db = (1 / m) * np.sum(predictions - y)

            self.w -= self.learning_rate * dw
            self.b -= self.learning_rate * db

    def predict(self, X):
        return self.w * X + self.b

