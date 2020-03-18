import numpy as np
import matplotlib.pyplot as plt
import data

# stabilni softmax
def stable_softmax(x):
    exp_x_shifted = np.exp(x - np.max(x))
    probs = exp_x_shifted / np.sum(exp_x_shifted)
    return probs

def logreg_train(X, Y_):

    C = max(Y_) + 1
    N = X.shape[0]
    D = X.shape[1]

    # Y_ = Y_.reshape((-1, 1))
    
    W = np.reshape([np.random.randn() for _ in range(C*D)], (C, D))
    b = np.array([np.random.randn() for _ in range(C)])

    param_niter = 100_000
    param_delta = 0.1

    for i in range(param_niter):

        # eksponencirane klasifikacijske mjere
        # pri računanju softmaksa obratite pažnju
        # na odjeljak 4.1 udžbenika
        # (Deep Learning, Goodfellow et al)!
        scores = X @ W.T + b    # N x C
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

        # derivacije komponenata gubitka po mjerama
        dL_ds = probs - Y_mat     # N x C

        # gradijenti parametara
        grad_W = dL_ds.T @ X / N    # C x D (ili D x C)
        grad_b = np.sum(dL_ds.T, axis=1)    # C x 1 (ili 1 x C)

        # poboljšani parametri  
        W += -param_delta * grad_W
        b += -param_delta * grad_b

    return W, b

def logreg_classify(X, W, b):
    N = len(X)
    
    # eksponencirane klasifikacijske mjere
    scores = X @ W.T + b    # N x C
    expscores = np.exp(scores - np.max(scores, axis=1).reshape(N, 1)) # N x C
    
    # nazivnik sofmaksa
    sumexp = np.sum(expscores, axis=1).reshape(N, 1)    # N x 1

    # vjerojatnosti razreda 
    probs = expscores / sumexp
    return probs


def eval_perf_multi(Y, Y_):
    C = len(np.unique(Y))
    N = len(Y)

    confusion_matrix = np.zeros((C, C))
    for y, y_ in zip(Y, Y_):
        confusion_matrix[y_, y] += 1

    diagonal = np.diag(confusion_matrix)

    accuracy = np.trace(confusion_matrix) / N
    precision = diagonal / np.sum(confusion_matrix, axis=0)
    recall = diagonal / np.sum(confusion_matrix, axis=1)
    return accuracy, precision, recall


def logreg_decfun(W, b):
    return lambda X: logreg_classify(X, W, b)[np.arange(len(X)), 1]


if __name__=="__main__":
    np.random.seed(99)

    # get the training dataset
    X, Y_ = data.sample_gauss_2d(3, 42)

    # train the model
    W, b = logreg_train(X, Y_)

    # evaluate the model on the training dataset
    probs = logreg_classify(X, W, b)
    Y = np.argmax(probs, axis=1)

    # report performance
    accuracy, precision, recall = eval_perf_multi(Y, Y_)
    print('accuracy', accuracy)
    print('precision', precision)
    print('recall', recall)
    
    # graph the decision surface
    decfun = logreg_decfun(W, b)
    bbox=(np.min(X, axis=0), np.max(X, axis=0))
    data.graph_surface(decfun, bbox, offset=0.5)

    # graph the data points
    data.graph_data(X, Y_, Y, special=[])

    # show the plot
    plt.show()
    