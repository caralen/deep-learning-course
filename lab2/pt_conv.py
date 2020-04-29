import os
import time
import math

import numpy as np
import matplotlib.pyplot as plt
import skimage as ski
import skimage.io

from pathlib import Path

import torch
from torch import nn
from torch import optim 
import torch.nn.functional as F
from torchvision.transforms import ToTensor, Normalize, Compose
from torchvision.datasets import MNIST

DATA_DIR = Path(__file__).parent / 'datasets' / 'MNIST'
SAVE_DIR = Path(__file__).parent / 'out' / 'mnist'

config = {}
config['max_epochs'] = 5
config['batch_size'] = 50
config['save_dir'] = SAVE_DIR
config['weight_decay'] = 1e-3
config['lr_policy'] = {1:1e-1, 3:1e-2, 5:1e-3, 7:1e-4}

 
class CovolutionalModel(nn.Module):
  def __init__(self, in_channels, conv1_width, conv2_width, fc1_width, class_count):
    super().__init__()

    self.conv1 = nn.Conv2d(in_channels, conv1_width, kernel_size=5, stride=1, padding=2, bias=True)
    self.pool1 = nn.MaxPool2d(2)
    self.conv2 = nn.Conv2d(conv1_width, conv2_width, kernel_size=5, stride=1, padding=2, bias=True)
    self.pool2 = nn.MaxPool2d(2)
    self.fc1 = nn.Linear(conv2_width*7*7, fc1_width, bias=True)
    self.fc_logits = nn.Linear(fc1_width, class_count, bias=True)

    self.reset_parameters()

  def reset_parameters(self):
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
    return logits

def draw_conv_filters(epoch, step, layer, save_dir):
  w = layer.weight.data.detach().numpy()
  num_filters = w.shape[0]
  C = w.shape[1]
  k = w.shape[2]
  
  w -= w.min()
  w /= w.max()
  border = 1
  cols = 8
  rows = math.ceil(num_filters / cols)
  width = cols * k + (cols-1) * border
  height = rows * k + (rows-1) * border

  for i in range(C):
    img = np.zeros([height, width])
    for j in range(num_filters):
      r = int(j / cols) * (k + border)
      c = int(j % cols) * (k + border)
      img[r:r+k,c:c+k] = w[j,i]
    filename = 'torch_conv1_epoch_%02d_step_%06d_input_%03d.png' % (epoch, step, i)
    ski.io.imsave(os.path.join(save_dir, filename), img)


def save_show_loss(loss_list, save_dir):
  filename = 'conv_losses.txt'
  f=open(os.path.join(save_dir, filename), 'w')
  for loss in loss_list:
      f.write(str(loss)+'\n')
  f.close()

  plt.plot(loss_list, marker='o')
  plt.title('Loss over epochs for CNN')
  plt.xlabel('epoch')
  plt.ylabel('loss')
  plt.show()


def train(model, trainloader, validloader):
  lr_policy = config['lr_policy']
  save_dir = config['save_dir']
  max_epochs = config['max_epochs']
  weight_decay = config['weight_decay']

  losses = []
  num_examples = len(trainloader.dataset)
  batch_size = trainloader.batch_size
  criterion = nn.CrossEntropyLoss()

  for epoch in range(1, max_epochs+1):
    cnt_correct = 0.0
    if epoch in lr_policy:
      optimizer = optim.SGD(model.parameters(), lr=lr_policy[epoch], weight_decay=weight_decay)

    for i, (images, labels) in enumerate(iter(trainloader), 1):
      logits = model.forward(images)
      loss = criterion(logits, labels)
      loss.backward()
      optimizer.step()
      optimizer.zero_grad()

      yp = logits.argmax(dim=1)
      cnt_correct += (yp == labels).sum()

      if (i == 1) or (i % 100 == 0):
        print("epoch %d, step %d/%d, batch loss = %.2f" % (epoch, i*batch_size, num_examples, loss))
      if i % 200 == 0:
        draw_conv_filters(epoch, i*batch_size, net.conv1, save_dir)
      if i > 0 and i % 500 == 0:
        print("Train accuracy = %.2f" % (cnt_correct / ((i+1)*batch_size) * 100))
    
    losses.append(loss.detach().numpy())
    evaluate("Validation", validloader, net, criterion)
  save_show_loss(losses, save_dir)


def evaluate(name, validloader, net, criterion):
  print("\nRunning evaluation: ", name)
  num_examples = len(validloader.dataset)
  assert num_examples % validloader.batch_size == 0
  num_batches = num_examples // validloader.batch_size
  cnt_correct = 0.0
  loss_avg = 0.0

  with torch.no_grad():
    for i, (images, labels) in enumerate(validloader):
      logits = net.forward(images)
      loss_val = criterion(logits, labels)
      loss_avg += loss_val

      yp = logits.argmax(dim=1)
      cnt_correct += (yp == labels).sum()

  valid_acc = cnt_correct / num_examples * 100
  loss_avg /= num_batches
  print(name + " accuracy = %.2f" % valid_acc)
  print(name + " avg loss = %.2f\n" % loss_avg)


if __name__ == "__main__":
  train_len = 55000
  transform = Compose([ToTensor(), Normalize([0.5], [0.5]),])
  traindata = MNIST(DATA_DIR, download=True, train=True, transform=transform)
  train_set, val_set = torch.utils.data.random_split(traindata, [train_len, len(traindata.data)-train_len])
  test_set = MNIST(DATA_DIR, download=False, train=False, transform=transform)
  trainloader = torch.utils.data.DataLoader(train_set, batch_size=config['batch_size'], shuffle=True)
  validloader = torch.utils.data.DataLoader(val_set, batch_size=config['batch_size'], shuffle=True)
  testloader = torch.utils.data.DataLoader(test_set, batch_size=config['batch_size'], shuffle=False)

  net = CovolutionalModel(in_channels = 1, conv1_width = 16, conv2_width = 32, fc1_width = 512, class_count = 10)
  train(net, trainloader, validloader)
  evaluate('Test', testloader, net, nn.CrossEntropyLoss())

