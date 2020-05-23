import torch
import torch.nn as nn

class RNN(nn.Module):
    def __init__(self, embedding, params):
        super().__init__()
        self.embedding = embedding
        self.params = params

        if params['cell_name'] == 'rnn':
            self.rnn = nn.RNN(input_size=300, hidden_size=params['hidden_size'], num_layers=params['num_layers'])
        elif params['cell_name'] == 'gru':
            self.rnn = nn.GRU(input_size=300, hidden_size=params['hidden_size'], num_layers=params['num_layers'])
        elif params['cell_name'] == 'lstm':
            self.rnn = nn.LSTM(input_size=300, hidden_size=params['hidden_size'], num_layers=params['num_layers'])
        else:
            raise AttributeError('Wrong cell name')

        self.fc1 = nn.Linear(params['hidden_size'], 150)
        self.fc2 = nn.Linear(150, 1)

    def forward(self, x):
        h = torch.transpose(x, 1, 0)
        h = self.embedding(h)

        if self.params['cell_name'] == 'lstm':
            _, (h_n, _) = self.rnn(h)
        else:
            _, h_n = self.rnn(h)

        h = h_n[-1]
        h = self.fc1(h)
        h = torch.relu(h)
        h = self.fc2(h)
        return h.flatten()

class Baseline(nn.Module):
    def __init__(self, embedding):
        super().__init__()
        self.embedding = embedding
        self.fc1 = nn.Linear(300, 150)
        self.fc2 = nn.Linear(150, 150)
        self.fc3 = nn.Linear(150, 1)

    def forward(self, x):
        h = self.embedding(x)
        h = torch.mean(h.float(), dim=1)
        h = self.fc1(h)
        h = torch.relu(h)
        h = self.fc2(h)
        h = torch.relu(h)
        h = self.fc3(h)
        return h.flatten()