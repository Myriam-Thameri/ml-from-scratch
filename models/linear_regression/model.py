import numpy as np
from optimizers.adaptive import AdaptiveSGD
class LinearRegression:
  def __init__(self, learning_rate=0.01,verbose=False,max_iters=100000, patience = 10 , toleration = 1e-6):
    self.learning_rate = learning_rate
    self.losses = []
    self.b = 0.0
    self.verbose=verbose
    self.max_iters = max_iters
    self.optimizer = AdaptiveSGD(lr=learning_rate)
    self.patience = patience
    self.toleration = toleration



  def scale(self,X):
    return (X - self.mean) / (self.std + 1e-8)

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
    X = self.scale(X)
    best_loss = float("inf")
    wait = 0
    iteration = 0

    while iteration < self.max_iters:

        y_pred = X @ self.w + self.b
        error = y_pred - y
        loss = np.mean(error ** 2)
        self.losses.append(loss)
        dw = (1/n_samples) * (X.T @ error)
        db = (1/n_samples) * np.sum(error)
        self.w, self.b = self.optimizer.update(self.w,self.b,dw,db,iteration,loss)

        # improvement check
        if best_loss - loss > self.toleration:
            best_loss = loss
            wait = 0
        else:
            wait += 1

        if wait >= self.patience:
            print("Converged: early stopping")
            break

        iteration += 1
    return self

  def predict(self,X):
    X = (X - self.mean) / self.std
    return X @ self.w + self.b
  

  




