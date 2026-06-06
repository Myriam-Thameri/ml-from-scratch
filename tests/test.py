import numpy as np
from models.linear_regression.model import LinearRegression
from metrics.regression import mean_absolute_error, root_mean_squared_error
from sklearn import linear_model
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


model = LinearRegression(learning_rate=0.01, verbose=True)
model.fit(X,y)
predictions = model.predict(X)
print("Predictions:", predictions)
mae = mean_absolute_error(y, predictions)
rmse = root_mean_squared_error(y, predictions)

print(f"Mean Absolute Error: ${mae:,.2f}")
print(f"Root Mean Squared Error: ${rmse:,.2f}")


sklearn_model = linear_model.LinearRegression()
sklearn_model.fit(X,y)
sklearn_predictions = sklearn_model.predict(X)
print(sklearn_predictions)
sklearn_mae = mean_absolute_error(y,sklearn_predictions)
sklearn_rmse = root_mean_squared_error(y,sklearn_predictions)
print(f"Mean Absolute Error: ${sklearn_mae:,.2f}")
print(f"Root Mean Squared Error: ${sklearn_rmse:,.2f}")