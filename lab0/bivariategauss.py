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

        np.random.seed(42)

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


if __name__=="__main__":
    G=Random2DGaussian()
    X=G.get_sample(100)
    plt.scatter(X[:,0], X[:,1])
    plt.show()