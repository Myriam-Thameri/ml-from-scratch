import numpy as np

class LinearRegression:
  def __init__(self, n_iters = 1000, learning_rate=0.01):
    self.learning_rate = learning_rate
    self.n_iters = n_iters
    self.losses = []
    self.b = 0.0

  def fit(self,X,y):
    if(len(X.shape) != 2):
      raise ValueError("X must be a 2D array")
    if(len(y.shape) != 1):
      raise ValueError("y must be a 1D array")
    if(X.shape[0] != y.shape[0]):
      raise ValueError("X and y must have the same number of samples")
    n_samples, n_features = X.shape
    self.w = np.zeros(n_features)
    #Feature standarization (Z-score normalization)
    self.mean = np.mean(X, axis=0)
    self.std = np.std(X, axis=0)
    #To avoid division by zero in case of constant features
    self.std[self.std == 0] = 1
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
  

  




