import numpy as np
import itertools
import collections
from collections import Counter, OrderedDict
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.utils.data as data


GLOVE_PATH = '/data/sst_glove_6b_300d.txt'
TRAIN_PATH = '/data/sst_train_raw.csv'
VALID_PATH = '/data/sst_valid_raw.csv'
TEST_PATH = '/data/sst_test_raw.csv'


class Vocab:
    
    def __init__(self, frequencies, max_size, min_freq):
        # frequencies je vec dict
        # jel sortiran?
        # jel bitno?
        # treba pazit na ove prve 2 pizdarije kod filtriranja i sortiranja

        # filtrirat za frekvenciju
        d = {key: value for (key, value) in frequencies.items() if value >= min_freq or key in ['<PAD>', '<UNK']}
        
        # sortirat
        # jel to uopce potrebno?
        sorted_frequencies = OrderedDict(Counter(d).most_common())
                
        # filtrirat jos za duzinu (slice)
        sorted_frequencies = itertools.islice(sorted_frequencies.items(), 0, max_size)

        # uzet kljuceve
        sorted_keys = sorted_frequencies.keys()
        
        self.stoi = dict(zip(sorted_keys, range(len(sorted_keys)))) 
        self.itos = {k:v for v,k in self.stoi.items()}

    def encode(self, seq):
        return [self.stoi[el] for el in seq]

    def decode(self, seq):
        return [self.itos[el] for el in seq]


@dataclass
class Instance:
    text: str
    label: str

class NLPDataset(data.Dataset):

    def __init__(self, instances):
        super().__init__()
        self.instances = instances
        self.D = 300

    def __getitem__(self, data_index):
        # TODO
        # treba vracat numerikaliziranu verziju podataka
        return self.instances[data_index]

    def generate_emb_matrix(self):
        N = len(vocab.stoi)
        return np.random.normal(0, 1, (N, self.D))

    def from_distrib(self):
        matrix = generate_emb_matrix()
        return torch.nn.Embedding.from_pretrained(matrix, padding_idx=0, freeze=False)

    def from_file(self, path):
        # TODO
        # ispada da tu jos pamtis vokabular
        # to bi trebalo ovisit o tome jel train ili ne
        self.instances = []
        with open(path, 'r') as f:
            for line in f:
                arr = line.split(', ')
                self.instances.append(Instance(arr[0], arr[1]))
    
    def from_file2(self, path):
        matrix = generate_emb_matrix()
        glove = {}

        with open(path, 'r') as f:
            for line in f:
                arr = line.split()
                key = arr[0]
                value = arr[1:-1]
                glove[key] = value

        for (k, v) in self.itos.items():
            if k == 0:
                matrix[k] = np.zeros(self.D)

            if v in glove.keys():
                matrix[k] = np.array(glove[v])
        
        return torch.nn.Embedding.from_pretrained(matrix, padding_idx=0, freeze=True)


def collate_fn(batch):
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
    return texts, labels, lengths


def main():
    instances = []
    with open(TRAIN_PATH, 'r') as f:
        for line in f:
            arr = line.split(', ')
            instances.append(Instance(arr[0], arr[1]))

    train_dataset = NLPDataset(instances)
    instance_text, instance_label = train_dataset.instances[3]
    print(f"Text: {instance_text}")
    print(f"Label: {instance_label}")
    # >>> Text: ['yet', 'the', 'act', 'is', 'still', 'charming', 'here']
    # >>> Label: positive

    # TODO
    # sto je s izgradnjom vokabulara?
    # trebao bih razredu Vocab predat dict frekvencija rijeci
    # jel to sam moram izgradit?
    text_vocab = Vocab(frequencies, max_size=-1, min_freq=0)
    print(len(text_vocab.itos))
    # >>> 14806

    print(f"Numericalized text: {text_vocab.encode(instance_text)}")
    print(f"Numericalized label: {label_vocab.encode(instance_label)}")
    # >>> Numericalized text: tensor([189,   2, 674,   7, 129, 348, 143])
    # >>> Numericalized label: tensor(0)

if __name__ == 'main':
    main()