import os
import pickle
import numpy as np

import torch
from torch import nn, optim
from collections import OrderedDict

class CifarConvolutionalModel(nn.Module):
  def __init__(self, num_channels):
    super().__init__()

    self.features = nn.Sequential(
      nn.Conv2d(num_channels, 16, kernel_size=5, stride=1, padding=2, bias=True),
      nn.MaxPool2d(3, 2),
      nn.ReLU(inplace=True),
      nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2, bias=True),
      nn.MaxPool2d(3, 2),
      nn.ReLU(inplace=True),
    )
    self.classifier = nn.Sequential(
      nn.Linear(1568, 256),
      nn.ReLU(inplace=True),
      nn.Linear(256, 128),
      nn.ReLU(inplace=True),
      nn.Linear(128, 10),
    )

  def forward(self, x):
    x = self.features(x)
    x = torch.flatten(x, 1)
    x = self.classifier(x)
    return x

def shuffle_data(data_x, data_y):
  indices = np.arange(data_x.shape[0])
  np.random.shuffle(indices)
  shuffled_data_x = np.ascontiguousarray(data_x[indices])
  shuffled_data_y = np.ascontiguousarray(data_y[indices])
  return shuffled_data_x, shuffled_data_y

def unpickle(file):
  fo = open(file, 'rb')
  dict = pickle.load(fo, encoding='latin1')
  fo.close()
  return dict

def train(model, train_x, train_y, valid_x, valid_y):
  max_epochs = 5
  batch_size = 50
  num_examples = train_x.shape[0]
  num_batches = num_examples // batch_size

  criterion = nn.CrossEntropyLoss()
  optimizer = optim.SGD(model.parameters(), lr=1e-3, weight_decay=1e-3)
  scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

  for epoch in range(1, max_epochs+1):

    permutation_idx = torch.randperm(num_examples)
    train_x = train_x[permutation_idx]
    train_y = train_y[permutation_idx]

    for i in range(num_batches):

      batch_x = train_x[i*batch_size:(i+1)*batch_size, :]
      batch_y = train_y[i*batch_size:(i+1)*batch_size]

      logits = model.forward(batch_x)
      loss = criterion(logits, batch_y)
      loss.backward()
      optimizer.step()
      optimizer.zero_grad()
      scheduler.step()

      if i % 5 == 0:
        print("epoch %d, step %d/%d, batch loss = %.2f" % (epoch, i*batch_size, num_examples, loss))

      if i % 10 == 0:
        print(eval_perf_multi(np.argmax(logits.detach().numpy(), axis=1), batch_y.detach().numpy()))

def evaluate(x, y, net, criterion):
  print("\nRunning evaluation: ", name)
  batch_size = config['batch_size']
  num_examples = x.shape[0]
  assert num_examples % batch_size == 0
  num_batches = num_examples // batch_siz

  with torch.no_grad():
    for i in range(num_batches):
      batch_x = x[i*batch_size:(i+1)*batch_size, :]
      batch_y = y[i*batch_size:(i+1)*batch_size]
      logits = net.forward(batch_x)
      loss_val = criterion(logits, batch_y)
      loss_avg += loss_val
      yp = logits.argmax(dim=1).detach().numpy()
      yt = batch_y.detach().numpy()

      # To bas nema smisla da radim tak za svaki batch posebno
      eval_perf_multi(yp, yt)

def eval_perf_multi(yp, yt):
  pr = []
  n = max(yt)+1
  M = np.bincount(n * yt + yp, minlength=n*n).reshape(n, n)
  for i in range(n):
      tp_i = M[i, i]
      fn_i = np.sum(M[i, :]) - tp_i
      fp_i = np.sum(M[:, i]) - tp_i
      tn_i = np.sum(M) - fp_i - fn_i - tp_i
      recall_i = tp_i / (tp_i + fn_i)
      precision_i = tp_i / (tp_i + fp_i)
      pr.append((recall_i, precision_i))
  accuracy = np.trace(M)/np.sum(M)
  return accuracy, pr, M

DATA_DIR = 'lab2/datasets/cifar-10-batches-py/'

img_height = 32
img_width = 32
num_channels = 3
num_classes = 10

train_x = np.ndarray((0, img_height * img_width * num_channels), dtype=np.float32)
train_y = []
for i in range(1, 6):
  subset = unpickle(os.path.join(DATA_DIR, 'data_batch_%d' % i))
  train_x = np.vstack((train_x, subset['data']))
  train_y += subset['labels']
train_x = train_x.reshape((-1, num_channels, img_height, img_width)).transpose(0,2,3,1)
train_y = np.array(train_y)

subset = unpickle(os.path.join(DATA_DIR, 'test_batch'))
test_x = subset['data'].reshape((-1, num_channels, img_height, img_width)).transpose(0,2,3,1).astype(np.float32)
test_y = np.array(subset['labels'])

valid_size = 5000
train_x, train_y = shuffle_data(train_x, train_y)
valid_x = train_x[:valid_size, ...]
valid_y = train_y[:valid_size, ...]
train_x = train_x[valid_size:, ...]
train_y = train_y[valid_size:, ...]
data_mean = train_x.mean((0,1,2))
data_std = train_x.std((0,1,2))

train_x = (train_x - data_mean) / data_std
valid_x = (valid_x - data_mean) / data_std
test_x = (test_x - data_mean) / data_std

train_x = train_x.transpose(0,3,1,2)
valid_x = valid_x.transpose(0,3,1,2)
test_x = test_x.transpose(0,3,1,2)

train_x = torch.from_numpy(train_x)
valid_x = torch.from_numpy(valid_x)
test_x = torch.from_numpy(test_x)
train_y = torch.from_numpy(train_y)
valid_y = torch.from_numpy(valid_y)
test_y = torch.from_numpy(test_y)

net = CifarConvolutionalModel(train_x.shape[1])
train(net, train_x, train_y, valid_x, valid_y)