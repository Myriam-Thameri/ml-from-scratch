from base import Optimizer

class AdaptiveSGD(Optimizer):
    def __init__(self, lr = 0.01):
        self.lr = lr
        self.prev_loss = float("inf")
        
    def update(self, w , b , dw , db ,  loss = None):
        lr = self.lr 
        
        if loss is not None and loss > self.prev_loss : 
            lr *= 0.05
            
        self.prev_loss = loss
        
        w -= lr * dw
        b -= lr * db
        
        return w ,b 
    