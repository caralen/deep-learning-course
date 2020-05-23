import torch
import torch.nn as nn

class RNN(nn.Module):
    def __init__(self, embedding):
        super().__init__()
        self.embedding = embedding
        self.gru = nn.GRU(input_size=300, hidden_size=150, num_layers=2)
        self.fc1 = nn.Linear(150, 150, bias=True)
        self.fc2 = nn.Linear(150, 1, bias=True)

    def forward(self, x):
        h = torch.transpose(x, 1, 0)
        h = self.embedding(h)
        _, h_n = self.gru(h)
        h = h_n[-1]
        h = self.fc1(h)
        h = torch.relu(h)
        h = self.fc2(h)
        return h.flatten()

class Baseline(nn.Module):
    def __init__(self, embedding):
        super().__init__()
        self.embedding = embedding
        self.fc1 = nn.Linear(300, 150, bias=True)
        self.fc2 = nn.Linear(150, 150, bias=True)
        self.fc3 = nn.Linear(150, 1, bias=True)

    def forward(self, x):
        h = self.embedding(x)
        h = torch.mean(h.float(), dim=1)
        h = self.fc1(h)
        h = torch.relu(h)
        h = self.fc2(h)
        h = torch.relu(h)
        h = self.fc3(h)
        return h.flatten()