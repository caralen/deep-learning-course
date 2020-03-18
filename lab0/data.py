import numpy as np
import matplotlib.pyplot as plt
import math

class Random2DGaussian():

    def __init__(self):
        super().__init__()

        minx=0
        maxx=10
        miny=0
        maxy=10
        cov_scaler = 5

        mux = (maxx - minx) * np.random.random_sample() + minx
        muy = (maxy - miny) * np.random.random_sample() + miny
        self.mu = [mux, muy]

        eigvalx = (np.random.random_sample()*(maxx - minx)/cov_scaler)**2
        eigvaly = (np.random.random_sample()*(maxy - miny)/cov_scaler)**2
        D = np.array([[eigvalx, 0], [0, eigvaly]])

        fi = 2 * math.pi * np.random.random_sample()
        R = np.array([[math.cos(fi), -math.sin(fi)], [math.sin(fi), math.cos(fi)]])

        self.cov_mat = R.T @ D @ R


    def get_sample(self, n):
        return np.random.multivariate_normal(self.mu, self.cov_mat, n)


def sample_gauss_2d(C, N):
    Y = []

    for i in range(C):
        G=Random2DGaussian()
        if i == 0:
            X = G.get_sample(N)
        else:
            X = np.vstack((X, G.get_sample(N)))
        Y += [i]*N
    return X, np.array(Y)

def sample_gmm_2d(ncomponents, nclasses, nsamples):
    # create the distributions and groundtruth labels
    Gs=[]
    Ys=[]
    for i in range(ncomponents):
        Gs.append(Random2DGaussian())
        Ys.append(np.random.randint(nclasses))

    # sample the dataset
    X = np.vstack([G.get_sample(nsamples) for G in Gs])
    Y_= np.hstack([[Y]*nsamples for Y in Ys])
    return X, Y_


def eval_perf_binary(Y,Y_):
    Y = np.array(Y)
    Y_ = np.array(Y_)

    tp = np.sum(np.logical_and(Y==Y_, Y==1))
    fp = np.sum(np.logical_and(Y!=Y_, Y==1))
    fn = np.sum(np.logical_and(Y!=Y_, Y==0))
    tn = np.sum(np.logical_and(Y==Y_, Y==0))

    accuracy = (tp + tn) / len(Y)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    return accuracy, precision, recall


def eval_AP(Yr):
    numerator = 0
    denominator = np.sum(Yr)

    for i in range(len(Yr)):
        Y = [0]*i + [1]*(len(Yr)-i)
        _, precision, _ = eval_perf_binary(Y, Yr)
        numerator += precision * Yr[i]

    return numerator / denominator


def graph_surface(function, rect, offset=0.5, width=256, height=256):
    """Creates a surface plot (visualize with plt.show)

    Arguments:
    function: surface to be plotted
    rect:     function domain provided as:
                ([x_min,y_min], [x_max,y_max])
    offset:   the level plotted as a contour plot

    Returns:
    None
    """

    lsw = np.linspace(rect[0][1], rect[1][1], width) 
    lsh = np.linspace(rect[0][0], rect[1][0], height)
    xx0, xx1 = np.meshgrid(lsh, lsw)
    grid = np.stack((xx0.flatten(), xx1.flatten()), axis=1)

    #get the values and reshape them
    values = function(grid).reshape((width, height))

    # fix the range and offset
    delta = offset if offset else 0
    maxval=max(np.max(values)-delta, - (np.min(values)-delta))

    # draw the surface and the offset
    plt.pcolormesh(xx0, xx1, values, vmin=delta-maxval, vmax=delta+maxval)

    if offset != None:
        plt.contour(xx0, xx1, values, colors='black', levels=[offset])


def graph_data(X,Y_, Y, special=[]):
    """Creates a scatter plot (visualize with plt.show)

    Arguments:
        X:       datapoints
        Y_:      groundtruth classification indices
        Y:       predicted class indices
        special: use this to emphasize some points

    Returns:
        None
    """
    # colors of the datapoint markers
    palette=([0.5,0.5,0.5], [1,1,1], [0.2,0.2,0.2])
    colors = np.tile([0.0,0.0,0.0], (Y_.shape[0],1))
    for i in range(len(palette)):
        colors[Y_==i] = palette[i]

    # sizes of the datapoint markers
    sizes = np.repeat(20, len(Y_))
    sizes[special] = 40

    # draw the correctly classified datapoints
    good = (Y_==Y)
    plt.scatter(X[good,0], X[good,1], c=colors[good], s=sizes[good], marker='o', edgecolors='black')

    # draw the incorrectly classified datapoints
    bad = (Y_!=Y)
    plt.scatter(X[bad,0], X[bad,1], c=colors[bad], s=sizes[bad], marker='s', edgecolors='black')


def my_dummy_decision(X):
    scores = X[:,0] + X[:,1] - 5
    return scores

if __name__=="__main__":
    np.random.seed(100)

    # get the training dataset
    X,Y_ = sample_gmm_2d(4, 2, 30)
    # X, Y_ = sample_gauss_2d(2, 100)

    # get the class predictions
    Y = my_dummy_decision(X)>0.5

    # graph the decision surface
    rect=(np.min(X, axis=0), np.max(X, axis=0))
    graph_surface(my_dummy_decision, rect, offset=0)

    # graph the data points
    graph_data(X, Y_, Y)

    # show the results
    plt.show()