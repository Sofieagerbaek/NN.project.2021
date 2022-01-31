import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot as plt

train_features = torch.load(snakemake.input[0])
train_labels = torch.load(snakemake.input[1])

val_features = torch.load(snakemake.input[2])
val_labels = torch.load(snakemake.input[3])

train_features =  torch.unsqueeze(train_features, 1)
val_features = torch.unsqueeze(val_features, 1)

train_data = TensorDataset(train_features, train_labels)
val_data = TensorDataset(val_features, val_labels)

train_loader = DataLoader(dataset=train_data,batch_size=2000,shuffle=True)
val_loader = DataLoader(dataset = val_data, batch_size=2000)

#Helper functions
def make_train_step(model, loss_fn, optimizer):
	def perform_train_step(x, y):
		model.train()
		y_pred = model(x)
		loss = loss_fn(y_pred, y)
		loss.backward()
		optimizer.step()
		optimizer.zero_grad()
		return loss.item()

	return perform_train_step

def make_validation_step(model, loss_fn):
	def perform_validation_step(x, y):
		model.eval()
		y_pred = model(x)
		loss = loss_fn(y_pred, y)
		return loss.item()

	return perform_validation_step

def mini_batch_descent(data_loader, step):
	losses = []
	for x_batch, y_batch in data_loader:
		mini_batch_loss = step(x_batch, y_batch)
		losses.append(mini_batch_loss)
	loss = np.mean(losses)
	return loss


#CNN class with arguments for parameter tuning
class conv_para_network(torch.nn.Module):

  def __init__(self, input_size, num_channels, conv_kernel_size, stride, pool_kernel_size, h1_size, padding=1):
    super(conv_para_network, self).__init__()
    self.conv1 = torch.nn.Conv1d(1, num_channels, conv_kernel_size, stride, padding) 
    conv1_size = ((input_size-conv_kernel_size+2*padding)//stride)+1
    self.bnorm1 = torch.nn.BatchNorm1d(num_channels)
    self.pool = torch.nn.MaxPool1d(kernel_size=pool_kernel_size)
    self.flatten = torch.nn.Flatten(start_dim=1)
    h1_in = (conv1_size//pool_kernel_size)*num_channels
    self.h1 = torch.nn.Linear(h1_in, h1_size)
    self.bnorm2 = torch.nn.BatchNorm1d(h1_size)
    self.output_layer = torch.nn.Linear(h1_size, 1)
    self.dropout = torch.nn.Dropout(0.3)
  
  def forward(self, x):
    x = self.bnorm1(self.conv1(x))
    x = torch.nn.functional.relu(self.pool(x))
    x = self.dropout(x)
    x = self.flatten(x)
    x = torch.nn.functional.relu(self.bnorm2(self.h1(x)))
    x = self.dropout(x)
    y = self.output_layer(x)

    return y

#Set parameters
lr = 1e-4
input_size = 804
h1_size = 25
pool_kernel_size = 2
n_epochs = 15
num_channels = 16

#Kernel size + stride
kernel_sizes = [8,12,32]
kernel_size_val_losses = []
strides = [4,16,24]


for kernel_size in kernel_sizes:
  stride_losses_train = []
  stride_losses_val = []

  for stride in strides:
    conv_kernel_size = kernel_size
    conv_stride = stride

    model = conv_para_network(input_size, num_channels, conv_kernel_size, conv_stride, pool_kernel_size, h1_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss = nn.MSELoss(reduction="mean")

    train_step = make_train_step(model, loss, optimizer)
    val_step = make_validation_step(model, loss)

    train_losses = []
    val_losses = []

    for epoch in range(n_epochs):
      train_loss = mini_batch_descent(train_loader, train_step)
      train_losses.append(train_loss)

      with torch.no_grad():
      	val_loss = mini_batch_descent(val_loader, val_step)
      	val_losses.append(val_loss)

    stride_losses_train.append(train_losses)
    stride_losses_val.append(val_losses)

  kernel_size_val_losses.append(stride_losses_val)


n = 0
for i in range(len(kernel_sizes)):
  plt.figure()
  for j in range(len(strides)):
    plt.plot(kernel_size_val_losses[i][j], label="stride: "+str(strides[j]))
  plt.legend()
  plt.title("Kernel size: "+str(kernel_sizes[i]))
  plt.savefig(snakemake.output[n])
  plt.close()
  n += 1


#filters

filters = [8,24,32,60]
filter_losses = []

for filt in filters:
  num_channels = filt
  conv_kernel_size = 12
  stride = 8

  model = conv_para_network(input_size, num_channels, conv_kernel_size, stride, pool_kernel_size, h1_size)
  loss = nn.MSELoss(reduction="mean")
  optimizer = torch.optim.Adam(model.parameters(), lr=lr)

  train_step = make_train_step(model, loss, optimizer)

  train_losses = []

  for epoch in range(n_epochs):
    train_loss = mini_batch_descent(train_loader, train_step)
    train_losses.append(train_loss)

  filter_losses.append(train_losses)

plt.plot(filter_losses[0], label ="8")
plt.plot(filter_losses[1], label="24")
plt.plot(filter_losses[2], label="32")
plt.plot(filter_losses[3], label="60")
plt.legend()
plt.savefig(snakemake.output[-2])
plt.close()


# #Weight-decay / L2 penalty
wds = [0.001,0.01,0.1,1]
wd_losses = []

for wd in wds:
  num_channels = 24
  conv_kernel_size = 12
  stride = 8

  model = conv_para_network(input_size, num_channels, conv_kernel_size, stride, pool_kernel_size, h1_size)
  loss = nn.MSELoss(reduction="mean")
  optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

  train_step = make_train_step(model, loss, optimizer)

  train_losses = []

  for epoch in range(n_epochs):
    train_loss = mini_batch_descent(train_loader, train_step)
    train_losses.append(train_loss)

  wd_losses.append(train_losses)

plt.plot(wd_losses[0], label ="wd = 0.001")
plt.plot(wd_losses[1], label="wd = 0.01")
plt.plot(wd_losses[2], label="wd = 0.1")
plt.plot(wd_losses[3], label="wd = 1")
plt.legend()
plt.savefig(snakemake.output[-1])
