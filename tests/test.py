import numpy as np
from models.linear_regression.model import LinearRegression
from metrics.regression import mean_absolute_error, root_mean_squared_error
import matplotlib as plt

#Housing prices
X = np.array([
    [1200, 3, 20],
    [1500, 4, 15],
    [1800, 4, 10],
    [2000, 5, 8],
    [2300, 5, 5]
])

y = np.array([
    200000,
    250000,
    320000,
    400000,
    480000
])

model = LinearRegression(n_iters=1000, learning_rate=0.01, verbose=True)
model.fit(X,y)
predictions = model.predict(X)
print("Predictions:", predictions)
mae = mean_absolute_error(y, predictions)
rmse = root_mean_squared_error(y, predictions)

print(f"Mean Absolute Error: ${mae:,.2f}")
print(f"Root Mean Squared Error: ${rmse:,.2f}")