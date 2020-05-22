import numpy as np
import itertools
import collections
from collections import Counter, OrderedDict
from dataclasses import dataclass
from pathlib import Path
import csv

import torch
import torch.nn as nn
# import torch.utils.data as data
from torch.utils.data import DataLoader, Dataset


GLOVE_PATH = Path(__file__).parent / 'data/sst_glove_6b_300d.txt'
TRAIN_PATH = Path(__file__).parent / 'data/sst_train_raw.csv'
VALID_PATH = Path(__file__).parent / 'data/sst_valid_raw.csv'
TEST_PATH = Path(__file__).parent / 'data/sst_test_raw.csv'


class Vocab:
    
    def __init__(self, frequencies, max_size, min_freq, labels=False):

        # filtrirat za frekvenciju
        filtered_freq = {key: value for (key, value) in frequencies.items() if value >= min_freq}
        
        # sortirat
        max_size = max_size if max_size != -1 else len(filtered_freq)
        words = [w for (w, cnt) in Counter(filtered_freq).most_common(max_size)]
        
        if not labels:
            words = ['<PAD>', '<UNK>'] + words
        
        self.stoi = dict(zip(words, range(len(words)))) 
        self.itos = {k:v for v,k in self.stoi.items()}

    def encode(self, seq):
        if type(seq) is str:
            return torch.tensor([self.stoi[seq]])
        return torch.tensor([self.stoi[el] for el in seq])

    def decode(self, seq):
        return torch.tensor([self.itos[el] for el in seq])

def generate_embedding_matrix(vocab, rand=False):
    D = 300
    N = len(vocab.stoi)
    matrix = np.random.normal(0, 1, (N, D))

    if rand:
        return torch.nn.Embedding.from_pretrained(matrix, padding_idx=0, freeze=False)

    glove = {}
    with open(GLOVE_PATH, 'r') as f:
        for line in f:
            arr = line.split()
            key = arr[0]
            value = arr[1:]
            glove[key] = value

    for (k, v) in vocab.itos.items():
        if k == 0:
            matrix[k] = np.zeros(D)

        if v in glove.keys():
            matrix[k] = np.array(glove[v])

    return torch.nn.Embedding.from_pretrained(torch.tensor(matrix), padding_idx=0, freeze=True)


@dataclass
class Instance:
    text: str
    label: list

class NLPDataset(Dataset):

    def __init__(self, instances, text_vocab, label_vocab):
        super().__init__()
        self.instances = instances
        self.text_vocab = text_vocab
        self.label_vocab = label_vocab

    def __getitem__(self, data_index):
        instance = self.instances[data_index]
        return self.text_vocab.encode(instance.text), self.label_vocab.encode(instance.label)

    def __len__(self):
        return len(self.instances)

    @staticmethod
    def from_file(path, text_vocab=None, label_vocab=None):
        instances = []
        with open(path, 'r') as f:
            for line in f:
                text, label = line.strip().split(', ')
                instances.append(Instance(text.split(), label))

        if text_vocab == None:
            all_text = []
            all_labels = []
            for instance in instances:
                all_text += instance.text
                all_labels += [instance.label]
                if instance.label == 'positive':
                    print()

            text_vocab = Vocab(Counter(all_text), max_size=-1, min_freq=0)
            label_vocab = Vocab(Counter(all_labels), max_size=-1, min_freq=0, labels=True)
        
        return NLPDataset(instances, text_vocab, label_vocab)


def pad_tensor(tensor, max_len, pad_index=0):
    seq = tensor.numpy().tolist()
    tensor_size = len(seq)
    if tensor_size == max_len:
        return seq
    return seq + [pad_index]*(max_len-tensor_size)
    # return torch.cat((tensor, torch.tensor([pad_index]*(max_len-tensor_size))))


def pad_collate_fn(batch, pad_index=0):
    """
    Arguments:
      Batch:
        list of Instances returned by `Dataset.__getitem__`.
    Returns:
      A tensor representing the input batch.
    """

    texts, labels = zip(*batch) # Assuming the instance is in tuple-like form
    lengths = torch.tensor([len(text) for text in texts]) # Needed for later
    # Process the text instances
    max_len = lengths.max().item()
    texts = torch.tensor([pad_tensor(text, max_len, pad_index) for text in texts])
    labels = torch.tensor([label.numpy().tolist() for label in labels]).flatten()
    return texts, labels, lengths


def load_dataset(batch_size):
    train_dataset = NLPDataset.from_file(TRAIN_PATH)
    valid_dataset = NLPDataset.from_file(VALID_PATH, train_dataset.text_vocab, train_dataset.label_vocab)
    test_dataset = NLPDataset.from_file(TEST_PATH, train_dataset.text_vocab, train_dataset.label_vocab)
    train_data_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, collate_fn=pad_collate_fn)
    valid_data_loader = DataLoader(dataset=valid_dataset, batch_size=batch_size, shuffle=False, collate_fn=pad_collate_fn)
    test_data_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, collate_fn=pad_collate_fn)
    return train_data_loader, valid_data_loader, test_data_loader


def main2():
    batch_size = 2 # Only for demonstrative purposes
    shuffle = False # Only for demonstrative purposes
    train_dataset = NLPDataset.from_file(TRAIN_PATH)
    train_data_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, 
                                shuffle=shuffle, collate_fn=pad_collate_fn)
    texts, labels, lengths = next(iter(train_data_loader))
    
    embedding = generate_embedding_matrix(train_dataset.text_vocab)
    x = embedding(texts)
    print(f"Texts: {x}")
    print(f"Labels: {labels}")
    print(f"Lengths: {lengths}")
    # >>> Texts: tensor([[   2,  554,    7, 2872,    6,   22,    2, 2873, 1236,    8,   96, 4800,
    #                     4,   10,   72,    8,  242,    6,   75,    3, 3576,   56, 3577,   34,
    #                     2022, 2874, 7123, 3578, 7124,   42,  779, 7125,    0,    0],
    #                 [   2, 2875, 2023, 4801,    5,    2, 3579,    5,    2, 2876, 4802,    7,
    #                     40,  829,   10,    3, 4803,    5,  627,   62,   27, 2877, 2024, 4804,
    #                     962,  715,    8, 7126,  555,    5, 7127, 4805,    8, 7128]])
    # >>> Labels: tensor([0, 0])
    # >>> Lengths: tensor([32, 34])

def main():
    train_dataset = NLPDataset.from_file(TRAIN_PATH)
    instance = train_dataset.instances[3]
    print(f"Text: {instance.text}")
    print(f"Label: {instance.label}")
    # >>> Text: ['yet', 'the', 'act', 'is', 'still', 'charming', 'here']
    # >>> Label: positive

    print(f"Numericalized text: {train_dataset.text_vocab.encode(instance.text)}")
    print(f"Numericalized label: {train_dataset.label_vocab.encode(instance.label)}")
    # >>> Numericalized text: tensor([189,   2, 674,   7, 129, 348, 143])
    # >>> Numericalized label: tensor(0)

    numericalized_text, numericalized_label = train_dataset[3]
    # Koristimo nadjačanu metodu indeksiranja
    print(f"Numericalized text: {numericalized_text}")
    print(f"Numericalized label: {numericalized_label}")
    # >>> Numericalized text: tensor([189,   2, 674,   7, 129, 348, 143])
    # >>> Numericalized label: tensor(0)

if __name__ == '__main__':
    main2()