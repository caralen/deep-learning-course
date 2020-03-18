import numpy as np
import matplotlib.pyplot as plt
import data

def binlogreg_train(X, Y_):
    '''
    Argumenti
        X:  podatci, np.array NxD
        Y_: indeksi razreda, np.array Nx1

    Povratne vrijednosti
        w, b: parametri logističke regresije
    '''
    N = X.shape[0]
    D = X.shape[1]

    Y_ = Y_.reshape((-1, 1))
    w = np.array([np.random.randn() for _ in range(D)]).reshape(-1, 1)
    b = 0

    param_niter = 100_000
    param_delta = 0.1

    for i in range(param_niter):

        # klasifikacijske mjere
        scores = np.dot(X, w) + b    # N x 1
        
        # vjerojatnosti razreda c_1
        probs = np.array(1.0/(1 + np.exp(-scores))).reshape(-1, 1)     # N x 1

        # gubitak
        loss = np.sum(-Y_*np.log(probs) - (1 - Y_)*np.log(1-probs)) / N
        
        # dijagnostički ispis
        if i % 1000 == 0:
            print("iteration {}: loss {}".format(i, loss))

        # derivacije gubitka po klasifikacijskim mjerama
        dL_dscores = probs - (Y_ == 1)    # N x 1
        
        # gradijenti parametara
        grad_w = np.transpose(np.dot(dL_dscores.T, X)) / N     # D x 1
        grad_b = np.sum(dL_dscores) / N    # 1 x 1

        # poboljšani parametri
        w += -param_delta * grad_w
        b += -param_delta * grad_b

    return w, b


def binlogreg_classify(X, w, b):
    '''
    Argumenti
        X:    podatci, np.array NxD
        w, b: parametri logističke regresije 

    Povratne vrijednosti
        probs: vjerojatnosti razreda c1
    '''

    scores = np.dot(X, w) + b    # N x 1
    return (1/(1 + np.exp(-scores))).flatten()


def binlogreg_decfun(w, b):
    return lambda X: binlogreg_classify(X, w, b)



if __name__=="__main__":
    np.random.seed(99)

    # get the training dataset
    X, Y_ = data.sample_gauss_2d(2, 42)

    # train the model
    w, b = binlogreg_train(X, Y_)

    # evaluate the model on the training dataset
    probs = binlogreg_classify(X, w,b)
    Y = (probs > 0.5).astype(int)

    # report performance
    accuracy, recall, precision = data.eval_perf_binary(Y, Y_)
    AP = data.eval_AP(Y_[probs.argsort()])
    print (accuracy, recall, precision, AP)
    
    # graph the decision surface
    decfun = binlogreg_decfun(w,b)
    bbox=(np.min(X, axis=0), np.max(X, axis=0))
    data.graph_surface(decfun, bbox, offset=0.5)

    # graph the data points
    data.graph_data(X, Y_, Y, special=[])

    # show the plot
    plt.show()