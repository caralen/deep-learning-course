import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path
import argparse
import os
import random
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.utils as utils

import data
from models import RNN, Baseline

SAVE_DIR = Path(__file__).parent / 'output/'


def train(model, data, optimizer, criterion, args):
    model.train()
    model.float()
    for batch_num, batch in enumerate(data):
        model.zero_grad()
        x, y, _ = batch
        logits = model.forward(x)
        loss = criterion(logits, y.float())
        loss.backward()
        utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()


def evaluate(model, data, criterion, name):
    model.eval()
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
        return accuracy, f1


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



def main_attention_test(args):
    chosen_params = {
        'cell_name': 'lstm',
        'hidden_size': 150,
        'num_layers': 2,
        'min_freq': 0,
        'lr': 1e-4,
        'dropout': 0,
        'freeze': True,
        'rand_emb': False,
        'attention': True
    }
    results = []

    for att in range(2):
        chosen_params['attention'] = True if att == 0 else False
        runs = 5
        acc_d = {}
        f1_d = {}

        for i in tqdm(range(runs)):
            train_dataset, valid_dataset, test_dataset = data.load_dataset(args.train_batch_size, args.test_batch_size, min_freq=chosen_params['min_freq'])
            embedding = data.generate_embedding_matrix(train_dataset.dataset.text_vocab, rand=chosen_params['rand_emb'], freeze=chosen_params['freeze'])

            model = RNN(embedding, chosen_params)
            criterion = nn.BCEWithLogitsLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=chosen_params['lr'])

            for epoch in range(args.epochs):
                print(f'******* epoch: {epoch+1} *******')
                train(model, train_dataset, optimizer, criterion, args)
                evaluate(model, valid_dataset, criterion, 'Validation')
            
            acc, f1 = evaluate(model, test_dataset, criterion, 'Test')
            acc_d['acc_' + 'run' + str(i)] = acc
            f1_d['f1_' + 'run' + str(i)] = f1

        mean = np.mean(list(acc_d.values()))
        std = np.std(list(acc_d.values()))
        acc_d['mean'] = mean
        acc_d['std'] = std

        mean = np.mean(list(f1_d.values()))
        std = np.std(list(f1_d.values()))
        f1_d['mean'] = mean
        f1_d['std'] = std

        results.append((acc_d, f1_d))

    with open(os.path.join(SAVE_DIR, 'attention.txt'), 'a') as f:
        print(f'', file=f)
        for idx, (acc, f1) in enumerate(results):
            print('[attention]' if idx == 0 else '[no attention]', file=f)
            print(acc, file=f)
            print(f1, file=f)


def main_hyperparam_optim(args):

    params = {
        'cell_name': ['lstm'],
        'hidden_size': [50, 150, 300],
        'num_layers': [2, 4, 5],
        'min_freq': [0, 100, 500],
        'lr': [1e-3, 1e-4, 1e-7],
        'dropout': [0, 0.4, 0.7],
        'freeze': [False, True],
        'rand_emb': [False, True],
        'attention': [False]
    }

    results = []
    for i in tqdm(range(10)):
        chosen_params = {k: random.choice(v) for (k, v) in params.items()}

        train_dataset, valid_dataset, test_dataset = data.load_dataset(args.train_batch_size, args.test_batch_size, min_freq=chosen_params['min_freq'])
        embedding = data.generate_embedding_matrix(train_dataset.dataset.text_vocab, rand=chosen_params['rand_emb'], freeze=chosen_params['freeze'])

        model = RNN(embedding, chosen_params)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=chosen_params['lr'])

        for epoch in range(args.epochs):
            print(f'******* epoch: {epoch+1} *******')
            train(model, train_dataset, optimizer, criterion, args)
            evaluate(model, valid_dataset, criterion, 'Validation')
        
        acc, f1 = evaluate(model, test_dataset, criterion, 'Test')
        result = dict(chosen_params)
        result['acc'] = acc
        result['f1'] = f1
        results.append

    with open(os.path.join(SAVE_DIR, 'params_search.txt'), 'a') as f:
        for result in results:
            print(result, file=f)




def main_cell_comparison(args):

    train_dataset, valid_dataset, test_dataset = data.load_dataset(args.train_batch_size, args.test_batch_size)
    embedding = data.generate_embedding_matrix(train_dataset.dataset.text_vocab)

    params = {
        'hidden_size': [50, 150, 300],
        'num_layers': [1, 2, 4],
        'dropout': [0.1, 0.4, 0.7],
        'bidirectional': [True, False]
    }

    for idx, (key, values) in enumerate(params.items()):
        fig, ax = plt.subplots(nrows=1, ncols=1)
        ax.set_title('Variable ' + key)
        for cell_name in tqdm(['rnn', 'lstm', 'gru']):
            results = []
            for i in range(len(values)):
                current_params = {k: v[i] if k==key else v[1] for (k,v) in params.items()}
                current_params['cell_name'] = cell_name

                model = RNN(embedding, current_params)
                model
                criterion = nn.BCEWithLogitsLoss()
                optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

                for epoch in range(args.epochs):
                    print(f'******* epoch: {epoch+1} *******')
                    train(model, train_dataset, optimizer, criterion, args)
                    evaluate(model, valid_dataset, criterion, 'Validation')

                result, _ = evaluate(model, test_dataset, criterion, 'Test')
                results.append(result)

            ax.plot(values, results, marker='o', label=cell_name)
        plt.legend(loc='best')
        plt.xlabel(key)
        plt.ylabel('accuracy')
        fig.savefig(os.path.join(SAVE_DIR, key + '.png'))
        plt.close(fig)


def main(args):

    train_dataset, valid_dataset, test_dataset = data.load_dataset(args.train_batch_size, args.test_batch_size)
    embedding = data.generate_embedding_matrix(train_dataset.dataset.text_vocab)
    model = RNN(embedding)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(args.epochs):
        print(f'******* epoch: {epoch} *******')
        train(model, train_dataset, optimizer, criterion, args)
        evaluate(model, valid_dataset, criterion, 'Validation')

    evaluate(model, test_dataset, criterion, 'Test')


def init(args):
    seed = args.seed
    np.random.seed(seed)
    torch.manual_seed(seed)

    main_attention_test(args)



if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('seed', type=int, help='seed for the random generator')
    parser.add_argument('epochs', type=int, help='number of epochs')
    parser.add_argument('train_batch_size', type=int, help='size of each batch in train')
    parser.add_argument('test_batch_size', type=int, help='size of each batch in test')
    parser.add_argument('clip', type=float, help='max norm of the gradients for gradient clipping')

    args = parser.parse_args()
    init(args)