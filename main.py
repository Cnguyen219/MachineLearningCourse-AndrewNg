import numpy as np
from linear_regression import LinearRegression

# Example data (house size vs price)
X = np.array([500, 800, 1200, 1500, 2000])
y = np.array([150000, 220000, 300000, 350000, 450000])

model = LinearRegression(learning_rate=0.00000001, iterations=1000)
model.fit(X, y)

prediction = model.predict(np.array([1800]))

print("Weight (w):", model.w)
print("Bias (b):", model.b)
print("Prediction for 1800 sq ft:", prediction[0])
