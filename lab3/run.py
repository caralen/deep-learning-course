import numpy as np
from pathlib import Path
import argparse

import torch
import torch.nn as nn
import torch.nn.utils as utils

import data
from models import RNN, Baseline

OUT_PATH = Path(__file__).parent / 'output/baseline.txt'


def train(model, data, optimizer, criterion, args):
    model.train()   # omogucava droupout - wut?! Sets the module in training mode!!!
    model.float()
    for batch_num, batch in enumerate(data):
        model.zero_grad()   # isto ko i optim.zero_grad() u ovom slucaju
        x, y, _ = batch    # ovo mozda nece ic ovak
        logits = model.forward(x)
        loss = criterion(logits, y.float())
        loss.backward()
        utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()

        if batch_num % 100 == 0:
            print(f'batch: {batch_num}, loss: {loss}')


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
        accuracy, f1 = [round(x*100, 3) for x in (accuracy, f1)]
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

    train_dataset, valid_dataset, test_dataset = data.load_dataset(args.train_batch_size, args.test_batch_size)
    embedding = data.generate_embedding_matrix(train_dataset.dataset.text_vocab)
    model = RNN(embedding, 'lstm')

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(args.epochs):
        print(f'******* epoch: {epoch} *******')
        train(model, train_dataset, optimizer, criterion, args)
        evaluate(model, valid_dataset, criterion, 'Validation')

    evaluate(model, test_dataset, criterion, 'Test')

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('seed', type=int, help='seed for the random generator')
    parser.add_argument('epochs', type=int, help='number of epochs')
    parser.add_argument('train_batch_size', type=int, help='size of each batch in train')
    parser.add_argument('test_batch_size', type=int, help='size of each batch in test')
    parser.add_argument('clip', type=float, help='max norm of the gradients for gradient clipping')

    args = parser.parse_args()
    print(args)
    main(args)