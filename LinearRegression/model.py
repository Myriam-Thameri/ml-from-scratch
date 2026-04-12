import numpy as np

class LinearRegression:
  def __init__(self, n_iters, learning_rate=0.01, b = 0.0):
    self.learning_rate = learning_rate
    self.n_iters = n_iters
    self.losses = []
    self.b = b

  def fit(self,X,y):
    n_samples, n_features = X.shape
    self.w = np.zeros(n_features)
    #Feature standarization (Z-score normalization)
    self.mean = np.mean(X, axis=0)
    self.std = np.std(X, axis=0)
    X = (X - self.mean) / self.std
    for _ in range(self.n_iters):
      #Prediction
      y_pred = X @ self.w + self.b

      #Gradient descent

      error = y_pred - y

      dw = (1/n_samples) * (X.T @ error)
      db = (1/n_samples) * np.sum(error)

      #Store loss
      self.losses.append(np.mean(error ** 2))

      #Update weight + bias
      self.w = self.w - (self.learning_rate * dw )
      self.b = self.b - (self.learning_rate * db)

  def predict(self,X):
    X = (X - self.mean) / self.std
    return X @ self.w + self.b
  

  




