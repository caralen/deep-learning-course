import numpy as np

def relu(x):
    return max(0, x)

def fcann2_train(X, Y_):

    C = max(Y_) + 1
    N = X.shape[0]
    D = X.shape[1]
    hidden_dim = 5

    # Y_ = Y_.reshape((-1, 1))
    
    W1 = np.reshape([np.random.randn() for _ in range(D*hidden_dim)], (D, hidden_dim))
    b1 = np.array([np.random.randn() for _ in range(hidden_dim)])
    W2 = np.reshape([np.random.randn() for _ in range(hidden_dim*C)], (hidden_dim, C))
    b2 = np.array([np.random.randn() for _ in range(C)])

    param_niter = 1e5
    param_delta = 0.05
    param_lambda = 1e-3

    for i in range(param_niter):
        S1 = X @ W1 + b1    # N x H
        H1 = S1 * (S1>0)    # N x H
        S2 = H1 @ W2 + b2   # N x C
        
        scores = S2.copy()    # N x C
        expscores = np.exp(scores - np.max(scores, axis=1).reshape(N, 1)) # N x C

        # nazivnik sofmaksa
        sumexp = np.sum(expscores, axis=1).reshape(N, 1)    # N x 1

        # logaritmirane vjerojatnosti razreda 
        probs = expscores / sumexp     # N x C
        logprobs = np.log(probs)  # N x C

        # gubitak
        loss  = - np.sum(logprobs[np.arange(N), Y_]) / N     # scalar

        # dijagnostički ispis
        if i % 10 == 0:
            print("iteration {}: loss {}".format(i, loss))

        Y_mat = np.zeros((N, C))
        Y_mat[np.arange(N), Y_] = 1

        dL_dS2 = probs - Y_mat     # N x C

        # gradijenti parametara
        grad_W2 = dL_dS2.T @ H1
        grad_b2 = dL_dS2

        dL_dS1 = dL_dS2 @ W1 @ np.diag(S1 > 0)     # N x C

        grad_W1 = dL_dS1.T @ X
        grad_b1 = dL_dS1

        # poboljšani parametri  
        W += -param_delta * grad_W
        b += -param_delta * grad_b

        

def fcann2_classify(X, W, b):
    pass

