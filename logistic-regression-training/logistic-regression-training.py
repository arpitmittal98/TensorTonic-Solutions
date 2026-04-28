import numpy as np

def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    """
    Train logistic regression via gradient descent.
    Return (w, b).
    """
    N, D = X.shape
    W = np.zeros(D)
    b = 0.0
    #y = y.ravel()
    
    for i in range(steps):
        y_pred = _sigmoid(np.dot(X,W)+b)
        #loss = -1/N * (np.sum(y*np.log(y_pred) +(1-y)*np.log(1-y_pred)))
        dw = 1/N * np.dot(X.T, y_pred-y)
        db = np.mean(y_pred - y)
        W = W - lr*dw
        b = b - lr*db

    return (W, float(b))