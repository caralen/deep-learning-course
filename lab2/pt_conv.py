import time
import numpy as np

from pathlib import Path

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision.datasets import MNIST

DATA_DIR = Path(__file__).parent / 'datasets' / 'MNIST'
SAVE_DIR = Path(__file__).parent / 'out'

config = {}
config['max_epochs'] = 8
config['batch_size'] = 50
config['save_dir'] = SAVE_DIR
config['weight_decay'] = 1e-3
config['lr_policy'] = {1:1e-1, 3:1e-2, 5:1e-3, 7:1e-4}

 
class CovolutionalModel(nn.Module):
  # _init__(self, in_channels, conv1_width, ..., fc1_width, class_count):
  def __init__(self, in_channels, conv1_width, fc1_width, class_count):
    super().__init__()

    self.conv1 = nn.Conv2d(in_channels, conv1_width, kernel_size=5, stride=1, padding=2, bias=True)
    self.pool1 = nn.MaxPool2d(2)
    self.conv2 = nn.Conv2d(conv1_width, 32, kernel_size=5, stride=1, padding=2, bias=True)
    self.pool2 = nn.MaxPool2d(2)
    self.fc1 = nn.Linear(32*7*7, fc1_width, bias=True)
    self.fc_logits = nn.Linear(fc1_width, class_count, bias=True)

  def reset_parmeters(self):
    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        nn.init.constant_(m.bias, 0)
      elif isinstance(m, nn.Linear) and m is not self.fc_logits:
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        nn.init.constant_(m.bias, 0)
    self.fc_logits.reset_parameters()

  def forward(self, x):
    h = self.conv1(x)
    h = self.pool1(h)
    h = torch.relu(h)
    h = self.conv2(h)
    h = self.pool2(h)
    h = torch.relu(h)
    h = h.view(h.shape[0], -1)
    h = self.fc1(h)
    h = torch.relu(h)
    logits = self.fc_logits(h)
    # F.softmax(logits)
    return logits

def train(model, train_x, train_y, valid_x, valid_y):

  lr_policy = config['lr_policy']
  batch_size = config['batch_size']
  max_epochs = config['max_epochs']
  weight_decay = config['weight_decay']

  num_examples = train_x.shape[0]
  num_batches = num_examples // batch_size

  criterion = nn.CrossEntropyLoss()
  # optimizer = optim.SGD(model.parameters(), lr=param_delta, weight_decay=weight_decay)

  # petlja učenja
  for epoch in range(1, max_epochs+1):

    if epoch in lr_policy:
      optimizer = optim.SGD(model.parameters(), lr=lr_policy[epoch], weight_decay=weight_decay)

    permutation_idx = torch.randperm(num_examples)
    train_x = train_x[permutation_idx]
    train_y = train_y[permutation_idx]

    for i in range(num_batches):

      batch_x = train_x[i*batch_size:(i+1)*batch_size, :]
      batch_y = train_y[i*batch_size:(i+1)*batch_size]

      # Prolaz unaprijed
      logits = model.forward(batch_x)

      # Dohvati gubitak
      loss = criterion(logits, batch_y)

      # računanje gradijenata
      loss.backward()

      # korak optimizacije
      optimizer.step()

      # Postavljanje gradijenata na nulu
      optimizer.zero_grad()

      if i % 5 == 0:
        print(f'epoch {epoch}, step: {i*batch_size}/{num_examples}, batch loss = {loss}')

def evaluate():
  pass

def dense_to_one_hot(y, class_count):
    return np.eye(class_count)[y]

if __name__ == "__main__":
  #np.random.seed(100)
  np.random.seed(int(time.time() * 1e6) % 2**31)

  ds_train, ds_test = MNIST(DATA_DIR, train=True, download=True), MNIST(DATA_DIR, train=False)
  train_x = ds_train.data.reshape([-1, 1, 28, 28]).numpy().astype(np.float) / 255
  train_y = ds_train.targets.numpy()
  train_x, valid_x = train_x[:55000], train_x[55000:]
  train_y, valid_y = train_y[:55000], train_y[55000:]
  test_x = ds_test.data.reshape([-1, 1, 28, 28]).numpy().astype(np.float) / 255
  test_y = ds_test.targets.numpy()
  train_mean = train_x.mean()
  train_x, valid_x, test_x = (x - train_mean for x in (train_x, valid_x, test_x))
  # train_y, valid_y, test_y = (dense_to_one_hot(y, 10) for y in (train_y, valid_y, test_y))

  train_x = torch.from_numpy(train_x)
  valid_x = torch.from_numpy(valid_x)
  test_x = torch.from_numpy(test_x)
  train_y = torch.from_numpy(train_y)
  valid_y = torch.from_numpy(valid_y)
  test_y = torch.from_numpy(test_y)

  net = CovolutionalModel(1, 16, 512, 10)
  train(net, train_x.float(), train_y, valid_x, valid_y)

