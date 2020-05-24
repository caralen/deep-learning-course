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
from train import train, evaluate
from models import RNN, Baseline

SAVE_DIR = Path(__file__).parent / 'output/'

params = {
    'cell_name': 'lstm',
    'hidden_size': 150,
    'num_layers': 2,
    'min_freq': 0,
    'lr': 1e-4,
    'dropout': 0,
    'freeze': True,
    'rand_emb': False,
    'attention': False
}



def embedding_baseline_test(args):
    chosen_params = dict(params)

    results = []

    for rand_emb in [True, False]:
        chosen_params['rand_emb'] = rand_emb
        train_dataset, valid_dataset, test_dataset = data.load_dataset(args.train_batch_size, args.test_batch_size, min_freq=chosen_params['min_freq'])
        embedding = data.generate_embedding_matrix(train_dataset.dataset.text_vocab, rand=chosen_params['rand_emb'], freeze=chosen_params['freeze'])

        result = {}
        for m in ['baseline', 'rnn']:
            if m == 'rnn':
                model = RNN(embedding, chosen_params)
            else:
                model = Baseline(embedding)

            criterion = nn.BCEWithLogitsLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

            for epoch in range(args.epochs):
                print(f'******* epoch: {epoch} *******')
                train(model, train_dataset, optimizer, criterion, args)
                evaluate(model, valid_dataset, criterion, 'Validation')

            acc, f1 = evaluate(model, test_dataset, criterion, 'Test')
            result[m + '_acc_rand_emb' + str(rand_emb)] = acc
            result[m + '_f1_rand_emb' + str(rand_emb)] = f1
        results.append(result)

    with open(os.path.join(SAVE_DIR, 'embedding_baseline.txt'), 'a') as f:
        for res in results:
            print(res, file=f)



def attention_test(args):
    chosen_params = dict(params)
    results = []

    for attention in [False, True]:
        chosen_params['attention'] = attention
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
            print('[no attention]' if idx == 0 else '[attention]', file=f)
            print(acc, file=f)
            print(f1, file=f)


def hyperparam_optim_test(args):

    var_params = {
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
        chosen_params = {k: random.choice(v) for (k, v) in var_params.items()}

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




def cell_comparison_test(args):

    train_dataset, valid_dataset, test_dataset = data.load_dataset(args.train_batch_size, args.test_batch_size)
    embedding = data.generate_embedding_matrix(train_dataset.dataset.text_vocab)

    var_params = {
        'hidden_size': [50, 150, 300],
        'num_layers': [1, 2, 4],
        'dropout': [0.1, 0.4, 0.7],
        'bidirectional': [True, False],
        'attention': [False]
    }

    for idx, (key, values) in enumerate(var_params.items()):
        fig, ax = plt.subplots(nrows=1, ncols=1)
        ax.set_title('Variable ' + key)
        for cell_name in tqdm(['rnn', 'lstm', 'gru']):
            results = []
            for i in range(len(values)):
                current_params = {k: v[i] if k==key else v[1] for (k,v) in var_params.items()}
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
    chosen_params = dict(params)

    train_dataset, valid_dataset, test_dataset = data.load_dataset(args.train_batch_size, args.test_batch_size)
    embedding = data.generate_embedding_matrix(train_dataset.dataset.text_vocab)
    model = RNN(embedding, params)

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

    attention_test(args)



if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('seed', type=int, help='seed for the random generator')
    parser.add_argument('epochs', type=int, help='number of epochs')
    parser.add_argument('train_batch_size', type=int, help='size of each batch in train')
    parser.add_argument('test_batch_size', type=int, help='size of each batch in test')
    parser.add_argument('clip', type=float, help='max norm of the gradients for gradient clipping')

    args = parser.parse_args()
    init(args)