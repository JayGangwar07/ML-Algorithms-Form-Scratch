import numpy as np

class SVM:

    def __init__(self,lr=0.001,lam=0.01,epochs=1000):
        self.lr = lr
        self.lam = lam
        self.epochs = epochs
        self.w = None
        self.b = None

    def fit(self,x,y):
        n_samples, n_features = x.shape
        y = np.where(y>=0,1,-1)
        self.w = np.zeros(n_features)
        self.b = 0

        for _ in range(self.epochs):
            for i,xi in enumerate(x):
                condition = y[i]*(np.dot(xi,self.w)-self.b)>=1
                if condition:
                    self.w -= self.lr*(2*self.lam*self.w)
                else:
                    self.w -= self.lr*(2*self.lam*self.w-np.dot(xi,y[i]))
                    self.b -= self.lr*y[i]

    def predict(self,x):
        ans = np.dot(x,self.w) - self.b
        return np.sign(ans)
