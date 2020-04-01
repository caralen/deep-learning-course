import numpy as np
import matplotlib.pyplot as plt
import data

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD


class PTLogreg(nn.Module):
    def __init__(self, D, C):
        """Arguments:
           - D: dimensions of each datapoint 
           - C: number of classes
        """
        super(PTLogreg, self).__init__()

        # inicijalizirati parametre (koristite nn.Parameter):
        # imena mogu biti self.W, self.b
        self.W = nn.Parameter(torch.rand(C, D, dtype=torch.double))
        self.b = nn.Parameter(torch.rand(C, 1))

    def forward(self, X):
        # unaprijedni prolaz modela: izračunati vjerojatnosti
        #   koristiti: torch.mm, torch.softmax
        scores = torch.mm(X, torch.t(self.W)) + torch.t(self.b)
        self.probs = F.softmax(scores)

    def get_loss(self, X, Yoh_):
        # formulacija gubitka
        #   koristiti: torch.log, torch.mean, torch.sum
        logprobs = torch.log(self.probs)
        loss = - torch.mean(torch.sum(logprobs * Yoh_, axis=1))
        # loss = - (torch.sum(torch.mean(logprobs, axis=1)) + param_lambda/2 * self.parameters()) / len(X)
        return loss


def train(model, X, Yoh_, param_niter, param_delta):
    """Arguments:
       - X: model inputs [NxD], type: torch.Tensor
       - Yoh_: ground truth [NxC], type: torch.Tensor
       - param_niter: number of training iterations
       - param_delta: learning rate
    """

    # inicijalizacija optimizatora
    optimizer = SGD(model.parameters(), lr=param_delta, weight_decay=1e-4)

    # petlja učenja
    for i in range(param_niter):
        # Prolaz unaprijed
        model.forward(X)

        # Dohvati gubitak
        loss = model.get_loss(X, Yoh_)

        # računanje gradijenata
        loss.backward()

        # korak optimizacije
        optimizer.step()

        # Postavljanje gradijenata na nulu
        optimizer.zero_grad()

        print(f'step: {i}, loss:{loss}')


def eval(model, X):
    """Arguments:
        - model: type: PTLogreg
        - X: actual datapoints [NxD], type: np.array
        Returns: predicted class probabilites [NxC], type: np.array
    """
    # ulaz je potrebno pretvoriti u torch.Tensor
    # izlaze je potrebno pretvoriti u numpy.array
    # koristite torch.Tensor.detach() i torch.Tensor.numpy()
    model.forward(torch.from_numpy(X))
    return model.probs.detach().numpy()


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


def pt_logreg_decfun(model):
    return lambda X: eval(model, X)[np.arange(len(X)), 1]


if __name__ == "__main__":
    # inicijaliziraj generatore slučajnih brojeva
    np.random.seed(42)

    C = 3
    N = 42

    # instanciraj podatke X i labele Yoh_
    X, Y_ = data.sample_gauss_2d(C, N)
    Yoh_ = F.one_hot(torch.from_numpy(Y_), C)

    # definiraj model:
    ptlr = PTLogreg(X.shape[1], Yoh_.shape[1])

    # nauči parametre (X i Yoh_ moraju biti tipa torch.Tensor):
    train(ptlr, torch.from_numpy(X), Yoh_, 1000, 0.05)

    # dohvati vjerojatnosti na skupu za učenje
    probs = eval(ptlr, X)
    Y = np.argmax(probs, axis=1)

    # ispiši performansu (preciznost i odziv po razredima)
    accuracy, precision, recall = data.eval_perf_multi(Y, Y_)
    print(f'accuracy: {accuracy}, precision: {precision}, recall: {recall}')

    # iscrtaj rezultate, decizijsku plohu
    decfun = pt_logreg_decfun(ptlr)
    bbox=(np.min(X, axis=0), np.max(X, axis=0))
    data.graph_surface(decfun, bbox, offset=0.5)
    data.graph_data(X, Y_, Y, special=[])

    # Prikazi
    plt.show()