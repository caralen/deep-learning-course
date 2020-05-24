import torch
import torch.nn as nn

class RNN(nn.Module):
    def __init__(self, embedding, params):
        super().__init__()
        self.embedding = embedding
        self.params = params

        cell_params = {
            'input_size': 300,
            'hidden_size': params['hidden_size'],
            'num_layers': params['num_layers'],
            'dropout': params['dropout']
        }

        if params['cell_name'] == 'rnn':
            self.rnn = nn.RNN(**cell_params)
        elif params['cell_name'] == 'gru':
            self.rnn = nn.GRU(**cell_params)
        elif params['cell_name'] == 'lstm':
            self.rnn = nn.LSTM(**cell_params)
        else:
            raise AttributeError('Wrong cell name')

        self.atten1 = nn.Linear(params['hidden_size'], int(params['hidden_size']/2))
        self.atten2 = nn.Linear(int(params['hidden_size']/2), 1)
        self.fc1 = nn.Linear(params['hidden_size'], params['hidden_size'])
        self.fc2 = nn.Linear(params['hidden_size'], 1)


    def attention_layer(self, h_n):
        h = self.atten1(h_n)
        h = torch.tanh(h)
        a = self.atten2(h)
        alpha = torch.softmax(a, dim=0)
        out = torch.sum(alpha*h_n, dim=0)
        return out


    def forward(self, x):
        h = torch.transpose(x, 1, 0)
        h = self.embedding(h)

        if self.params['cell_name'] == 'lstm':
            _, (h_n, _) = self.rnn(h)
        else:
            _, h_n = self.rnn(h)

        if self.params['attention']:
            h = self.attention_layer(h_n)
        else:
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