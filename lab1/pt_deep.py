import numpy as np
import matplotlib.pyplot as plt
import data

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD


class PTDeep(nn.Module):
    def __init__(self, layers, activation):
        """Arguments:
           - layers: list containing number of neurons for each layer
           - activation: non-linear activation function
        """
        super(PTDeep, self).__init__()

        self.weights = nn.ParameterList([])
        self.biases = nn.ParameterList([])
        self.activation = activation

        for i, layer in enumerate(layers):
            if i == 0:
                continue
            
            self.weights.append(nn.Parameter(torch.rand(layers[i-1], layer, dtype=torch.double)))
            self.biases.append(nn.Parameter(torch.rand(1, layer)))


    def forward(self, X, test=False):
        # unaprijedni prolaz modela: izračunati vjerojatnosti
        #   koristiti: torch.mm, torch.softmax

        S = torch.mm(X, self.weights[0]) + self.biases[0]

        # Batch normalization
        if ~test:
            S = (S - S.mean(dim=1).view(-1,1)) / torch.sqrt(S.std(dim=1).view(-1,1))

        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            if i == 0:
                continue

            H = self.activation(S)
            S = torch.mm(H, W) + b

            # Batch normalization
            if ~test:
                mean = S.mean(dim=1)
                std = S.std(dim=1)
                S = (S - S.mean(dim=1).view(-1,1)) / torch.sqrt(S.std(dim=1).view(-1,1))


        self.probs = F.softmax(S)

    def get_loss(self, X, Yoh_, param_lambda):
        # formulacija gubitka
        #   koristiti: torch.log, torch.mean, torch.sum
        sum_norms = 0
        for W in self.weights:
            sum_norms += torch.sum(torch.norm(W, p=2, dim=1))

        logprobs = torch.log(self.probs)
        loss = - torch.mean(torch.sum(logprobs * Yoh_, axis=1)) + param_lambda/(2*len(X)) * sum_norms
        return loss


    def count_params(self):
        total_params = 0
        for name, param in self.named_parameters():
            total_params += param.size()[0]*param.size()[1]
            print(f'name: {name}, size: {param.size()}')
        print('total parameters:', total_params)


def train(model, X, Yoh_, param_niter, param_delta, param_lambda=1e-3):
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
        loss = model.get_loss(X, Yoh_, param_lambda)

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
    model.forward(torch.from_numpy(X), test=True)
    return model.probs.detach().numpy()


def pt_deep_decfun(model):
    return lambda X: eval(model, X)[np.arange(len(X)), 1]


if __name__ == "__main__":
    # inicijaliziraj generatore slučajnih brojeva
    np.random.seed(100)

    config = [2, 10, 10, 2]

    # instanciraj podatke X i labele Yoh_
    X, Y_ = data.sample_gmm_2d(6, 2, 10)
    Yoh_ = F.one_hot(torch.from_numpy(Y_), config[-1])

    # definiraj model:
    ptdeep = PTDeep(config, torch.relu)

    # nauči parametre (X i Yoh_ moraju biti tipa torch.Tensor):
    train(ptdeep, torch.from_numpy(X), Yoh_, 10_000, 0.1)

    # dohvati vjerojatnosti na skupu za učenje
    probs = eval(ptdeep, X)
    Y = np.argmax(probs, axis=1)

    # ispiši performansu (preciznost i odziv po razredima)
    accuracy, pr, _ = data.eval_perf_multi(Y, Y_)
    print(f'accuracy: {accuracy}, precision: {pr[0]}, recall: {pr[1]}')
    
    # Ispiši imena i broj parametara
    ptdeep.count_params()

    # iscrtaj rezultate, decizijsku plohu
    decfun = pt_deep_decfun(ptdeep)
    bbox=(np.min(X, axis=0), np.max(X, axis=0))
    data.graph_surface(decfun, bbox, offset=0.5)
    data.graph_data(X, Y_, Y, special=[])

    # Prikaži
    plt.show()