from base import Optimizer
class SGDD(Optimizer):
    def __init__(self, lr = 0.01):
        self.lr = lr
        
    def update(self, w , b , dw , db , iteration):
        w -= self.lr * dw
        b -= self.lr * db
        return w , b