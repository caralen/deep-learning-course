import numpy as np
import torch
import torch.nn as nn


class Baseline(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.AveragePool1d(
            kernel_size=300, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False)
        self.fc1 = nn.Linear(300, 150, bias=True)
        self.fc2 = nn.Linear(150, 150, bias=True)
        self.fc3 = nn.Linear(150, 1, bias=True)

    def forward(self, x):
        h = self.pool(x)
        h = self.fc1(h)
        h = torch.relu(h)
        h = self.fc2(h)
        h = torch.relu(h)
        h = self.fc3(h)
        return h


def train(model, data, optimizer, criterion, args):
    model.train()   # omogucava droupout - wut?! Sets the module in training mode!!!
    for batch_num, batch in enumerate(data):
        model.zero_grad()   # isto ko i optim.zero_grad() u ovom slucaju
        x, y = batch    # ovo mozda nece ic ovak
        logits = model.forward(x)
        loss = criterion(logits, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()

        if batch_num % 1_000:
            print(f'batch: {batch_num}, loss: {loss}')


def evaluate(model, data, criterion, args):
    model.eval()    # Sets the model in eval mode
    with torch.no_grad():
        for batch_num, batch in enumerate(data):
            x, y = batch
            logits = model(x)
            loss = criterion(logits, y)
            # racunat confusion matrix i ostale metrike


def main(args):
    seed = args.seed
    np.random.seed(seed)
    torch.manual_seed(seed)

    train_dataset, valid_dataset, test_dataset = load_dataset(...)
    model = initialize_model(args, ...)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(args.epochs):
        train(model, train_dataset, optimizer, criterion)
        evaluate(model, valid_dataset, criterion, args)

    evaluate(model, test_dataset, criterion, args)
