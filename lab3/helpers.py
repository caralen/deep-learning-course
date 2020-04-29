import numpy as np
from collections import Counter, OrderedDict

class TextDataLoader():

    def __init__(self, batch_size = 50, sequence_length = 5):
        self.batch_size = batch_size
        self.sequence_length = sequence_length

    def preprocess(self, input_file):
        with open(input_file, "r") as f:
            data = f.read()
            
        # count and sort most frequent characters
        d = Counter(data)
        self.sorted_chars = OrderedDict(d.most_common()).keys()
        # self.sorted chars contains just the characters ordered descending by frequency
        self.char2id = dict(zip(self.sorted_chars, range(len(self.sorted_chars)))) 
        # reverse the mapping
        self.id2char = {k:v for v,k in self.char2id.items()}
        # convert the data to ids
        self.x = np.array(list(map(self.char2id.get, data)))

    def encode(self, sequence):
        return [self.char2id[el] for el in sequence]

    def decode(self, encoded_sequence):
        return [self.id2char[el] for el in encoded_sequence]

    def create_minibatches(self):
        character_length = self.batch_size * self.sequence_length
        self.num_batches = int(len(self.x) / character_length)
        assert self.num_batches >= 1

        self.batches = []
        for batch_i in range(self.num_batches):
            x = [self.x[i] for i in range(batch_i*character_length, (batch_i+1)*character_length)]
            y = [self.x[i+1] for i in range(batch_i*character_length, (batch_i+1)*character_length)]
            self.batches.append((x, y))

    def next_minibatch(self, new_epoch):
        if new_epoch:
            self.current_batch = 0
        else:
            self.current_batch += 1

        assert self.current_batch < self.num_batches
        batch_x, batch_y = self.batches[self.current_batch]
        return batch_x, batch_y


def main_preprocess():
    loader = TextDataLoader()
    loader.preprocess('lab3/data/selected_conversations.txt')
    enc = loader.encode(['a', 'l', 'e', 'n'])
    dec = loader.decode(enc)


def main_create_batches():
    loader = TextDataLoader()
    loader.preprocess('lab3/data/selected_conversations.txt')
    b = loader.create_minibatches()
    print(np.shape(b))

def main():
    loader = TextDataLoader()
    loader.preprocess('lab3/data/selected_conversations.txt')
    loader.create_minibatches()

    for epoch in range(3):
        for b in range(loader.num_batches):
            batch_x, batch_y = loader.next_minibatch(new_epoch = True if b == 0 else False)


if __name__ == '__main__':
    main()