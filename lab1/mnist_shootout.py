import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD
from torch.optim import Adam
import torchvision

import data

# **********************************************************************************************************
# Kompletno rjesenje je u Jupyter notebook-u mnist_shootout.ipynb jer je lakse rijesavati zadatak po zadatak
# **********************************************************************************************************

def whitening(weights, axis=0):
    mean = weights.mean(axis=axis)
    std = weights.std(axis=axis)
    return (weights - mean) / std

def generate_weights(rows, columns=1):
    return np.reshape([np.random.randn() for _ in range(rows*columns)], (rows, columns))


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
            
            self.weights.append(nn.Parameter(torch.from_numpy(whitening(generate_weights(layers[i-1], layer)))))
            self.biases.append(nn.Parameter(torch.from_numpy(whitening(generate_weights(1, layer), axis=1))))


    def forward(self, X, test=False):
        # unaprijedni prolaz modela: izračunati vjerojatnosti
        #   koristiti: torch.mm, torch.softmax

        S = torch.mm(X.double(), self.weights[0]) + self.biases[0]

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

        self.probs = torch.softmax(S - torch.max(S, 1)[0].view(-1, 1), 1, torch.float)
        self.logprobs = F.log_softmax(S)

        

    def get_loss(self, X, Yoh_):
        # formulacija gubitka
        #   koristiti: torch.log, torch.mean, torch.sum

        # logprobs = torch.log(self.probs)
        loss = - torch.mean(torch.sum(self.logprobs * Yoh_, axis=1))
        return loss


    def get_high_loss_data_indexes(self, X, Yoh_, top_k=5):
        # Pronalazi k indeksa slika za koje je gubitak najveci

        logprobs = torch.log(self.probs)
        arr = torch.sum(logprobs * Yoh_, axis=1).detach().numpy().ravel()
        return arr.argsort()[-top_k:][::-1]


    def count_params(self):
        total_params = 0
        for name, param in self.named_parameters():
            total_params += param.size()[0]*param.size()[1]
            print(f'name: {name}, size: {param.size()}')
        print('total parameters:', total_params)


def train(model, X, Yoh_, param_niter, param_delta, param_lambda=1e-4):
    """Arguments:
    - X: model inputs [NxD], type: torch.Tensor
    - Yoh_: ground truth [NxC], type: torch.Tensor
    - param_niter: number of training iterations
    - param_delta: learning rate
    """
    
    # inicijalizacija optimizatora
    optimizer = SGD(model.parameters(), lr=param_delta, weight_decay=param_lambda)

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
    model.forward(X, test=True)
    return model.probs.detach().numpy()


def pt_deep_decfun(model):
    return lambda X: eval(model, X)[np.arange(len(X)), 1]


def show_weight_matrix(model):
    # Ispis matrice težina za model konfiguracije [784, 10] 
    W = model.weights[0].detach().numpy()
    print(W)
    df = pd.DataFrame(W)
    display(df.corr())
    df.plot(subplots=True)
    plt.show()


def show_high_loss_pics(model, x_train, y_oh_train):
    D = x_train.shape[1] * x_train.shape[2]
    pic_index = model.get_high_loss_data_indexes(x_train.view(-1, D), y_oh_train, 3)

    for i in pic_index:
        plt.imshow(x_train[i,:,:], cmap = plt.get_cmap('gray'))
        plt.show()


def train_validation_split(x, y, test_size=0.2):
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=42)
    X_train = torch.from_numpy(X_train)
    X_val = torch.from_numpy(X_test)
    y_train = torch.from_numpy(y_train)
    y_val = torch.from_numpy(y_test)
    return X_train, X_val, y_train, y_val

def train_mb(model, X, Yoh_, param_epochs, param_batches, param_delta, param_lambda=1e-4):

    # inicijalizacija optimizatora
    optimizer = Adam(model.parameters(), lr=param_delta, weight_decay=param_lambda)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=1-1e-4)

    for epoch in range(param_epochs):

        # mijesaj podatke
        # podijeli u grupe
        idx = torch.randperm(X.size()[0])
        X = X[idx]
        Yoh_ = Yoh_[idx]

        batch_len = len(X) / param_batches

        for cur_batch in range(param_batches):

            batch_idx = torch.tensor(np.arange(cur_batch * batch_len, (cur_batch+1)*batch_len), dtype=torch.long)
            X_batch = X[batch_idx]
            Yoh_batch = Yoh_[batch_idx]

            # Prolaz unaprijed
            model.forward(X_batch)

            # Dohvati gubitak
            loss = model.get_loss(X_batch, Yoh_batch)

            # računanje gradijenata
            loss.backward()

            # korak optimizacije
            optimizer.step()

            # Postavljanje gradijenata na nulu
            optimizer.zero_grad()

            # Korak schedulera
            scheduler.step()

            print(f'batch: {cur_batch+1}/{param_batches}, epoch: {epoch+1}, loss:{loss}')




if __name__ == "__main__":
    # inicijaliziraj generatore slučajnih brojeva
    np.random.seed(100)

    config = [784, 100, 10]

    dataset_root = '/tmp/mnist'  # change this to your preference
    mnist_train = torchvision.datasets.MNIST(dataset_root, train=True, download=True)
    mnist_test = torchvision.datasets.MNIST(dataset_root, train=False, download=True)

    x_train, y_train = mnist_train.data, mnist_train.targets
    x_test, y_test = mnist_test.data, mnist_test.targets
    x_train, x_test = x_train.float().div_(255.0), x_test.float().div_(255.0)

    y_oh_train = F.one_hot(y_train, config[-1])
    y_oh_test = F.one_hot(y_test, config[-1])

    N = x_train.shape[0]
    D = x_train.shape[1] * x_train.shape[2]
    C = y_train.max().add_(1).item()

    model = PTDeep(config, torch.relu)
    train(model, x_train.view(-1, D), y_oh_train, 100, 0.1)
    # train_mb(model, x_train.view(-1, D), y_oh_train, 100, 100, 1e-4)

    # Prikazi slike koje najvise doprinose funkciji gubitka
    # show_high_loss_pics(model, x_train, y_oh_train)

    # dohvati vjerojatnosti na skupu za učenje
    probs = eval(model, x_test.view(-1, D))
    Y_pred = np.argmax(probs, axis=1)

    # ispiši performansu (preciznost i odziv po razredima)
    accuracy, pr, _ = data.eval_perf_multi(Y_pred, y_test)
    print(f'accuracy: {accuracy}, precision: {pr[0]}, recall: {pr[1]}')
    
    # Ispiši imena i broj parametara
    model.count_params()
    # show_weight_matrix(model)