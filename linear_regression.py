import numpy as np
class Linear_regression():
    def __init__(self,lr=0.001,n_iters=1000):
        self.lr=lr
        self.n_iters=n_iters
        weight=None
        bias=None
    def fit(self,x,y):
        n_samples,n_features=x.shape
        self.bias=0
        self.weight=np.zeros(n_features)
        for _ in range(self.n_iters):
            ypred=np.dot(x,self.weight)+self.bias
            dw=(1/n_samples)*np.dot((y-ypred),x)
            db=(1/n_samples)*np.sum(y-ypred)
            self.weight=-self.lr*dw
            self.bias=-self.lr*db

        
    def predict(self,x):
        ypred=np.dot(x,self.weight)+self.bias
        return ypred