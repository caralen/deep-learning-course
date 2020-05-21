import numpy as np
from pathlib import Path
import argparse

import torch
import torch.nn as nn

import data

OUT_PATH = Path(__file__).parent / 'output/baseline.txt'


class Baseline(nn.Module):
    def __init__(self, embedding):
        super().__init__()
        # kernel size je duljina recenice, tj te sekvence rijeci
        # sta nije da to ovisi o batchu?
        self.embedding = embedding
        self.pool = nn.AvgPool1d(kernel_size=300)
        self.fc1 = nn.Linear(300, 150, bias=True)
        self.fc2 = nn.Linear(150, 150, bias=True)
        self.fc3 = nn.Linear(150, 1, bias=True)

    def forward(self, x):
        h = embedding(x)
        h = torch.mean(x, dim=1)
        # h = torch.mean(x.float(), dim=1)
        # h = self.pool(x)
        h = self.fc1(h)
        h = torch.relu(h)
        h = self.fc2(h)
        h = torch.relu(h)
        h = self.fc3(h)
        return h


def train(model, data, optimizer, criterion, args):
    model.train()   # omogucava droupout - wut?! Sets the module in training mode!!!
    for batch_num, batch in enumerate(data):
        model.zero_grad()   # isto ko i optim.zero_grad() u ovom slucaju
        x, y, _ = batch    # ovo mozda nece ic ovak
        logits = model.forward(x)
        loss = criterion(logits, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()

        if batch_num % 1_000:
            print(f'batch: {batch_num}, loss: {loss}')


def evaluate(model, data, criterion, args):
    model.eval()    # Sets the model in eval mode
    with torch.no_grad():
        Y, Y_ = ([], [])
        for batch_num, batch in enumerate(data):
            x, y, _ = batch
            logits = model.forward(x)
            loss = criterion(logits, y)
            yp = torch.round(logits)

            Y += yp.detach().numpy().tolist()
            Y_ += y.detach().numpy().tolist()

        acc, (precision, recall), M = eval_perf_multi(np.array(Y), np.array(Y_))
        print(f'Accuracy: {acc}')

def eval_perf_multi(Y, Y_):
    pr = []
    n = max(Y_)+1
    M = np.bincount(n * Y_ + Y, minlength=n*n).reshape(n, n)
    for i in range(n):
        tp_i = M[i, i]
        fn_i = np.sum(M[i, :]) - tp_i
        fp_i = np.sum(M[:, i]) - tp_i
        tn_i = np.sum(M) - fp_i - fn_i - tp_i
        recall_i = tp_i / (tp_i + fn_i)
        precision_i = tp_i / (tp_i + fp_i)
        pr.append((recall_i, precision_i))

    acc = np.trace(M)/np.sum(M)
    return acc, pr, M


def main(args):
    seed = args.seed
    np.random.seed(seed)
    torch.manual_seed(seed)

    train_dataset, valid_dataset, test_dataset = data.load_dataset(args.batch_size)
    embedding = data.generate_embedding_matrix(train_dataset.dataset.text_vocab)
    # model = initialize_model(args, ...)
    model = Baseline(embeddings)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(args.epochs):
        train(model, train_dataset, optimizer, criterion, args)
        evaluate(model, valid_dataset, criterion, args)

    evaluate(model, test_dataset, criterion, args)

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('seed', type=int, help='seed for the random generator')
    parser.add_argument('epochs', type=int, help='number of epochs')
    parser.add_argument('batch_size', type=int, help='size of each batch')
    parser.add_argument('clip', type=int, help='')

    args = parser.parse_args()
    print(args)
    main(args)