import numpy as np

class LogisticRegression:
    def __init__(self,learning_rate=0.01,max_iters = 100000,patience = 10, tolerance = 1e-6):
        self.losses = []
        self.b = 0
        self.max_iters = max_iters
        self.learning_rate = learning_rate
        self.patience = patience
        self.tolerance = tolerance
        
    
    def sigmoid(self, value):
        return (1/(1+np.exp(-value)))
    
    def scale(self,X):
        return (X - self.mean) / (self.std + 1e-8)
    
    def binary_cross_entropy(self, y , y_pred):
        eps = 1e-6
        y_pred = np.clip(y_pred, eps, 1 - eps)
        loss = - np.mean(y * np.log(y_pred) + (1- y)*np.log(1-y_pred))
        return loss
    
    def fit(self, X, y, regularizer = None, lambda_ = 0.01):
        if(X.shape[0] != y.shape[0]):
            raise ValueError("X and y must have the same number of samples")
        n_samples, n_features = X.shape
        #Initilize weights
        self.w = np.zeros(n_features)
        #Z score
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X,axis=0)
        self.std[self.std == 0 ] = 1
        X = self.scale(X)
        best_loss = float("inf") 
        wait = 0
        iteration = 0
        while(iteration < self.max_iters):
            linear_output = np.dot(X,self.w) + self.b
            y_pred = self.sigmoid(linear_output)
            
            loss = self.binary_cross_entropy(y,y_pred)
            self.losses.append(loss)
            dw = (np.dot(X.T,(y_pred - y))) / n_samples
            db = np.sum(y_pred - y) / n_samples
            
            
            self.w -= self.learning_rate * dw
            self.b -= self.learning_rate * db
            
            if best_loss - loss > self.tolerance:
                best_loss = loss
                wait = 0
            else:
                wait += 1
            
            if wait > self.patience:
                print("The model converged")
                break
                
            iteration+=1
        return self
    
    
    def predict(self,X):
        X = self.scale(X)
        linear_output = np.dot(X, self.w) + self.b
        y_pred = self.sigmoid(linear_output)
        return (y_pred >= 0.5).astype(int)
        
        
        