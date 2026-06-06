from base import Optimizer
import numpy as np
class DecaySGD(Optimizer):
    def __init__(self, lr=0.01):
        self.lr = lr
        
        
    def update(self,w,b,dw,db,iteration):
        lr = self.lr / np.sqrt(iteration + 1)
        
        w -= lr * dw
        b -= lr * db
        
        return w,b