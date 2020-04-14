import numpy as np
import data
import matplotlib.pyplot as plt

def relu(x):
    return max(0, x)

def whitening(weights):
    mean = weights.mean(axis=0)
    std = weights.std(axis=0)
    return (weights - mean) / std

def generate_weights(rows, columns=1):
    return np.reshape([np.random.randn() for _ in range(rows*columns)], (rows, columns))

def fcann2_train(X, Y_):

    param_niter = 10_000
    param_delta = 0.05
    param_lambda = 1e-3

    C = max(Y_) + 1
    N = X.shape[0]
    D = X.shape[1]
    hidden_dim = 5
    
    W1 = whitening(generate_weights(D, hidden_dim)) # D x H
    b1 = whitening(generate_weights(hidden_dim))    # H x 1
    W2 = whitening(generate_weights(hidden_dim, C)) # H x C
    b2 = whitening(generate_weights(C))             # C x 1

    for i in range(param_niter):

        # 1) ***** Forward pass *****
        S1 = X @ W1 + b1.T      # N x H
        H1 = S1 * (S1>0)        # N x H
        S2 = H1 @ W2 + b2.T     # N x C
        
        scores = S2.copy()      # N x C
        expscores = np.exp(scores - np.max(scores, axis=1).reshape(N, 1)) # N x C

        # nazivnik sofmaksa
        sumexp = np.sum(expscores, axis=1).reshape(N, 1)    # N x 1

        # logaritmirane vjerojatnosti razreda 
        probs = expscores / sumexp     # N x C
        logprobs = np.log(probs)  # N x C

        # gubitak
        # loss  = - np.mean(logprobs[np.arange(N), Y_])     # scalar
        # loss  = - np.mean(logprobs[np.arange(N), Y_]) + 1/2 * param_lambda * np.mean(np.linalg.norm(W2))
        loss = np.mean(np.max(scores, axis=1) - scores[np.arange(N), Y_])

        # dijagnostički ispis
        if i % 10 == 0:
            print("iteration {}: loss {}".format(i, loss))

        Y_mat = np.zeros((N, C))
        Y_mat[np.arange(N), Y_] = 1


        # 2) ***** Backward pass *****
        dL_dS2 = probs - Y_mat                              # N x C

        grad_W2 = dL_dS2.T @ H1                             # C x H
        grad_b2 = np.sum(dL_dS2.T, axis=1).reshape(-1, 1)   # C x 1
        # grad_b2 = dL_dS2              # C x 1

        dL_dS1 = dL_dS2 @ W2.T * np.diag(S1 > 0)            # N x H

        grad_W1 = dL_dS1.T @ X                              # H x D
        grad_b1 = np.sum(dL_dS1.T, axis=1).reshape(-1, 1)   # H x 1

        
        # 2) ***** Update params *****
        W1 += -param_delta * grad_W1.T
        b1 += -param_delta * grad_b1
        W2 += -param_delta * grad_W2.T
        b2 += -param_delta * grad_b2

    return W1, b1, W2, b2
        

def fcann2_classify(X, W1, b1, W2, b2):
    N = len(X)

    S1 = X @ W1 + b1.T      # N x H
    H1 = S1 * (S1>0)        # N x H
    S2 = H1 @ W2 + b2.T     # N x C
    
    scores = S2.copy()      # N x C
    expscores = np.exp(scores - np.max(scores, axis=1).reshape(N, 1)) # N x C

    # nazivnik sofmaksa
    sumexp = np.sum(expscores, axis=1).reshape(N, 1)    # N x 1

    # logaritmirane vjerojatnosti razreda 
    probs = expscores / sumexp     # N x C

    return probs


def fcann2_decfun(W1, b1, W2, b2):
    return lambda X : fcann2_classify(X, W1, b1, W2, b2)[np.arange(len(X)), 1]


if __name__=="__main__":
    np.random.seed(100)

    # get data
    X, Y_ = data.sample_gmm_2d(6, 2, 10)

    # train the model
    W1, b1, W2, b2 = fcann2_train(X, Y_)

    # get the class predictions
    probs = fcann2_classify(X, W1, b1, W2, b2)
    Y = np.argmax(probs, axis=1)

    # # graph the decision surface
    decfun = fcann2_decfun(W1, b1, W2, b2)
    rect = (np.min(X, axis=0), np.max(X, axis=0))
    data.graph_surface(decfun, rect, offset=0)

    # # graph the data points
    data.graph_data(X, Y_, Y, special=[])

    plt.show()