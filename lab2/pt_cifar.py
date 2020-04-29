import os
import math
import pickle
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import nn, optim
from collections import OrderedDict

import skimage as ski
import skimage.io


SAVE_DIR = 'lab2/out/cifar/'
DATA_DIR = 'lab2/datasets/cifar-10-batches-py/'
MAX_EPOCHS = 1
BATCH_SIZE = 50


def conv_output_dim(img_w, kernel_size, stride):
  return (img_w - kernel_size)//stride + 1

class CifarConvolutionalModel(nn.Module):
  def __init__(self, img_w, in_channels, conv1_width, conv2_width, fc1_width, fc2_width, class_count):
    super().__init__()

    pool_kernel_size = 3
    pool_stride = 2
    pool_layers = 2

    self.features = nn.Sequential(
      nn.Conv2d(in_channels, conv1_width, kernel_size=5, stride=1, padding=2, bias=True),
      nn.MaxPool2d(pool_kernel_size, pool_stride),
      nn.ReLU(inplace=True),
      nn.Conv2d(conv1_width, conv2_width, kernel_size=5, stride=1, padding=2, bias=True),
      nn.MaxPool2d(pool_kernel_size, pool_stride),
      nn.ReLU(inplace=True),
    )

    for i in range(pool_layers):
      img_w = conv_output_dim(img_w=img_w, kernel_size=pool_kernel_size, stride=pool_stride)

    self.classifier = nn.Sequential(
      nn.Linear(conv2_width * img_w**2, fc1_width),
      nn.ReLU(inplace=True),
      nn.Linear(fc1_width, fc2_width),
      nn.ReLU(inplace=True),
      nn.Linear(fc2_width, class_count),
    )

  def forward(self, x):
    x = self.features(x)
    x = torch.flatten(x, 1)
    x = self.classifier(x)
    return x



def evaluate(name, x, y, net, criterion):
  print("\nRunning evaluation: ", name)
  
  batch_size = 50
  num_examples = x.shape[0]
  num_batches = num_examples // batch_size
  cum_loss = 0
  yp = []
  yt = []

  with torch.no_grad():
    for i in range(num_batches):
      batch_x = x[i*batch_size:(i+1)*batch_size, :]
      batch_y = y[i*batch_size:(i+1)*batch_size]

      logits = net.forward(batch_x)
      cum_loss += criterion(logits, batch_y).data.numpy()

      yp.append(logits.argmax(dim=1).detach().numpy())
      yt.append(batch_y.detach().numpy())

    loss = cum_loss/num_batches
    accuracy, pr, M = eval_perf_multi(np.reshape(yp, -1), np.reshape(yt, -1))
    print('loss: %.2f, accuracy: %.2f\n' % (loss, 100*accuracy))
    return loss, accuracy

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

def plot_training_progress(save_dir, data):
  fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16,8))

  linewidth = 2
  legend_size = 10
  train_color = 'm'
  val_color = 'c'

  num_points = len(data['train_loss'])
  x_data = np.linspace(1, num_points, num_points)
  ax1.set_title('Cross-entropy loss')
  ax1.plot(x_data, data['train_loss'], marker='o', color=train_color,
           linewidth=linewidth, linestyle='-', label='train')
  ax1.plot(x_data, data['valid_loss'], marker='o', color=val_color,
           linewidth=linewidth, linestyle='-', label='validation')
  ax1.legend(loc='upper right', fontsize=legend_size)
  ax2.set_title('Average class accuracy')
  ax2.plot(x_data, data['train_acc'], marker='o', color=train_color,
           linewidth=linewidth, linestyle='-', label='train')
  ax2.plot(x_data, data['valid_acc'], marker='o', color=val_color,
           linewidth=linewidth, linestyle='-', label='validation')
  ax2.legend(loc='upper left', fontsize=legend_size)
  ax3.set_title('Learning rate')
  ax3.plot(x_data, data['lr'], marker='o', color=train_color,
           linewidth=linewidth, linestyle='-', label='learning_rate')
  ax3.legend(loc='upper left', fontsize=legend_size)

  save_path = os.path.join(save_dir, 'training_plot.png')
  print('Plotting in: ', save_path)
  plt.savefig(save_path)


def draw_conv_filters(epoch, step, weights, save_dir):
  w = weights.copy()
  num_filters = w.shape[0]
  num_channels = w.shape[1]
  k = w.shape[2]
  assert w.shape[3] == w.shape[2]
  w = w.transpose(2,3,1,0)
  w -= w.min()
  w /= w.max()
  border = 1
  cols = 8
  rows = math.ceil(num_filters / cols)
  width = cols * k + (cols-1) * border
  height = rows * k + (rows-1) * border
  img = np.zeros([height, width, num_channels])
  for i in range(num_filters):
    r = int(i / cols) * (k + border)
    c = int(i % cols) * (k + border)
    img[r:r+k,c:c+k,:] = w[:,:,:,i]
  filename = 'epoch_%02d_step_%06d.png' % (epoch, step)
  ski.io.imsave(os.path.join(save_dir, filename), img)


def draw_image(img, mean, std):
  img = img.transpose(1,2,0)
  img *= std
  img += mean
  img = img.astype(np.uint8)
  ski.io.imshow(img)
  ski.io.show()


def show_k_highest_loss_images(model, x, y, data_mean, data_std, k=20):
  x_numpy = x.detach().numpy()
  criterion = nn.CrossEntropyLoss(reduction='none')

  with torch.no_grad():
    logits = model.forward(x)
    losses = criterion(logits, y)
    indices = torch.topk(losses, k)[1].detach().numpy()

    for i in indices:
      draw_image(x_numpy[i], data_mean, data_std)
      print('True class:', y[i].data)
      print('3 highest prob classes: ', torch.topk(logits[i, :], 3)[1].data)


def multiclass_hinge_loss(logits: torch.Tensor, target: torch.Tensor, delta=1.):  
  """
  Args:
    logits: torch.Tensor with shape (B, C), where B is batch size, and C is number of classes.
    target: torch.LongTensor with shape (B, ) representing ground truth labels.
    delta: Hyperparameter.
  Returns:
    Loss as scalar torch.Tensor.
  """
  target_onehot = torch.zeros(logits.shape, dtype=torch.bool)
  target_onehot.scatter_(1, target.reshape(-1,1), True)
  scores_hit = torch.masked_select(logits, target_onehot).reshape(-1,1)
  scores_miss = torch.masked_select(logits, ~target_onehot).reshape(-1, logits.shape[1]-1)
  return torch.sum(torch.max(torch.zeros(scores_miss.shape), scores_miss - scores_hit + delta), dim=1).mean()


def train(model, train_x, train_y, valid_x, valid_y):
  max_epochs = MAX_EPOCHS
  batch_size = BATCH_SIZE
  num_examples = train_x.shape[0]
  num_batches = num_examples // batch_size

  plot_data = {}
  plot_data['train_loss'] = []
  plot_data['valid_loss'] = []
  plot_data['train_acc'] = []
  plot_data['valid_acc'] = []
  plot_data['lr'] = []

  # criterion = multiclass_hinge_loss
  criterion = nn.CrossEntropyLoss()
  optimizer = optim.SGD(model.parameters(), lr=1e-1, weight_decay=1e-3)
  lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.1)

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

      if i % 100 == 0:
        print("epoch %d, step %d/%d, batch loss = %.2f" % (epoch, i*batch_size, num_examples, loss))

      if i % 200 == 0:
        draw_conv_filters(epoch, i, model.features[0].weight.detach().cpu().numpy(), SAVE_DIR)

    train_loss, train_acc = evaluate('Training', train_x, train_y, model, criterion)
    val_loss, val_acc = evaluate('Validation', valid_x, valid_y, model, criterion)

    plot_data['train_loss'] += [train_loss]
    plot_data['valid_loss'] += [val_loss]
    plot_data['train_acc'] += [train_acc]
    plot_data['valid_acc'] += [val_acc]
    plot_data['lr'] += [lr_scheduler.get_lr()]
    lr_scheduler.step()

  plot_training_progress(SAVE_DIR, plot_data)


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

train_x = torch.from_numpy(train_x.transpose(0,3,1,2))
valid_x = torch.from_numpy(valid_x.transpose(0,3,1,2))
test_x = torch.from_numpy(test_x.transpose(0,3,1,2))
train_y = torch.from_numpy(train_y)
valid_y = torch.from_numpy(valid_y)
test_y = torch.from_numpy(test_y)

net = CifarConvolutionalModel(img_w=img_width, in_channels=num_channels, conv1_width=16, 
  conv2_width=32, fc1_width=256, fc2_width=128, class_count=num_classes)

draw_conv_filters(0, 0, net.features[0].weight.detach().numpy(), SAVE_DIR)
train(net, train_x, train_y, valid_x, valid_y)

test_loss, test_acc = evaluate('Test', test_x, test_y, net, nn.CrossEntropyLoss())
show_k_highest_loss_images(net, test_x, test_y, data_mean, data_std)