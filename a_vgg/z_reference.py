import numpy as np

def softmax(X):
    exps = np.exp(X - np.max(X))
    return exps / np.sum(exps)

def cross_entropy(X,y):
    """
    X is the output from fully connected layer (num_examples x num_classes)
    y is labels (num_examples x 1)
    """
    m = y.shape[0]
    p = softmax(X)

    print(range(m))

    log_likelihood = -(y * np.log(p) ) 
    loss = np.sum(log_likelihood) / m
    return loss



x = np.random.randn(5,10)
y = np.random.randn(5,1)

tempss = cross_entropy(x,y)

print(tempss)


def delta_cross_entropy(X,y):
    """
    X is the output from fully connected layer (num_examples x num_classes)
    y is labels (num_examples x 1)
    """
    m = y.shape[0]
    grad = softmax(X)
    grad =  grad - y
    grad = grad/m
    return grad

tempss = delta_cross_entropy(x,y)

print(tempss.shape)
