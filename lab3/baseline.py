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
        # self.pool = nn.AvgPool1d(kernel_size=300)
        self.fc1 = nn.Linear(300, 150, bias=True)
        self.fc2 = nn.Linear(150, 150, bias=True)
        self.fc3 = nn.Linear(150, 1, bias=True)

    def forward(self, x):
        h = self.embedding(x)
        h = torch.mean(h.float(), dim=1)
        # h = self.pool(x)
        h = self.fc1(h)
        h = torch.relu(h)
        h = self.fc2(h)
        h = torch.relu(h)
        h = self.fc3(h)
        return h.flatten()


def train(model, data, optimizer, criterion, args):
    model.train()   # omogucava droupout - wut?! Sets the module in training mode!!!
    for batch_num, batch in enumerate(data):
        model.zero_grad()   # isto ko i optim.zero_grad() u ovom slucaju
        x, y, _ = batch    # ovo mozda nece ic ovak
        logits = model.forward(x)
        loss = criterion(logits, y.float())
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()

        # if batch_num % 1_000:
        #     print(f'batch: {batch_num}, loss: {loss}')


def evaluate(model, data, criterion, name):
    model.eval()    # Sets the model in eval mode
    with torch.no_grad():
        Y, Y_ = ([], [])
        for batch_num, batch in enumerate(data):
            x, y, _ = batch
            logits = model.forward(x)
            loss = criterion(logits, y.float())
            yp = torch.round(torch.sigmoid(logits)).int()

            Y += yp.detach().numpy().tolist()
            Y_ += y.detach().numpy().tolist()

        accuracy, f1, confusion_matrix = eval_perf_binary(np.array(Y), np.array(Y_))
        print(f'[{name}] accuracy: {accuracy}, f1: {f1}, confusion_matrix: {confusion_matrix}')


def eval_perf_binary(Y, Y_):
    tp = sum(np.logical_and(Y == Y_, Y_ == True))
    fn = sum(np.logical_and(Y != Y_, Y_ == True))
    tn = sum(np.logical_and(Y == Y_, Y_ == False))
    fp = sum(np.logical_and(Y != Y_, Y_ == False))
    recall = tp / (tp + fn)
    precision = tp / (tp + fp)
    accuracy = (tp + tn) / (tp+fn + tn+fp)
    f1 = 2 * (recall*precision) / (recall + precision)
    confusion_matrix = [[tp, fp], [fn, tn]]
    return accuracy, f1, confusion_matrix


def main(args):
    seed = args.seed
    np.random.seed(seed)
    torch.manual_seed(seed)

    train_dataset, valid_dataset, test_dataset = data.load_dataset(args.batch_size)
    embedding = data.generate_embedding_matrix(train_dataset.dataset.text_vocab)
    # model = initialize_model(args, ...)
    model = Baseline(embedding)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(args.epochs):
        train(model, train_dataset, optimizer, criterion, args)
        evaluate(model, valid_dataset, criterion, 'Validation')

    evaluate(model, test_dataset, criterion, 'Test')

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('seed', type=int, help='seed for the random generator')
    parser.add_argument('epochs', type=int, help='number of epochs')
    parser.add_argument('batch_size', type=int, help='size of each batch')
    parser.add_argument('clip', type=int, help='')

    args = parser.parse_args()
    print(args)
    main(args)