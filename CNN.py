import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split as split
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import copy
from sklearn.metrics import mean_squared_error

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


#CNN class
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
    self.dropout = torch.nn.Dropout(0.2)
  
  def forward(self, x):
    x = self.bnorm1(self.conv1(x))
    x = torch.nn.functional.relu(self.pool(x))
    x = self.dropout(x)
    x = self.flatten(x)
    x = torch.nn.functional.relu(self.bnorm2(self.h1(x)))
    x = self.dropout(x)
    y = self.output_layer(x)

    return y


#initializing
lr = 1e-4
input_size = train_features.shape[2]
num_channels = 32
conv_kernel_size = 48
stride = 4
h1_size = 150
pool_kernel_size = 2

model = conv_para_network(input_size, num_channels, conv_kernel_size, stride, pool_kernel_size, h1_size)
loss = nn.MSELoss(reduction="mean")
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.01)


train_step = make_train_step(model, loss, optimizer)
val_step = make_validation_step(model, loss)

#Actual training loop

n_epochs = 10
min_val_loss = 1

train_losses = []
val_losses = []

for epoch in range(n_epochs):
  train_loss = mini_batch_descent(train_loader, train_step)
  train_losses.append(train_loss)

  with torch.no_grad():
    val_loss = mini_batch_descent(val_loader, val_step)
    val_losses.append(val_loss)
    torch.save(model.state_dict(), snakemake.output[2])

torch.save(model.state_dict(), snakemake.output[1])

plt.plot(train_losses, label = "Training loss")
plt.plot(val_losses, label = "Validation loss")
plt.legend()
plt.savefig(snakemake.output[0])
