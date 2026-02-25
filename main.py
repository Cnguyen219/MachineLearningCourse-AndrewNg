import numpy as np
from linear_regression import LinearRegression

# Example training data (house size vs price)
X = np.array([1, 2, 3, 4])
y = np.array([300, 500, 700, 900])

model = LinearRegression(learning_rate=0.01, iterations=1000)

# Train the model
model.fit(X, y)

print("Learned weight (w):", model.w)
print("Learned bias (b):", model.b)

# This will make a prediction
print("Prediction for x=5:", model.predict(np.array([5])))
