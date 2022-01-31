import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, TensorDataset, DataLoader
import matplotlib.pyplot as plt

train_features = torch.load(snakemake.input[0])
train_labels = torch.load(snakemake.input[1])

val_features = torch.load(snakemake.input[2])
val_labels = torch.load(snakemake.input[3])

train_data = TensorDataset(train_features, train_labels)
val_data = TensorDataset(val_features, val_labels)

train_loader = DataLoader(dataset=train_data,batch_size=2000,shuffle=True)
val_loader = DataLoader(dataset = val_data, batch_size=2000)

class NeuralNet(torch.nn.Module):
	
	def __init__(self, input_len, h1_size, h2_size, h3_size, p):
		super(NeuralNet, self).__init__()
		self.h1 = torch.nn.Linear(input_len, h1_size)
		self.h2 = torch.nn.Linear(h1_size, h2_size)
		self.h3 = torch.nn.Linear(h2_size, h3_size)

		self.bn1 = torch.nn.BatchNorm1d(h1_size)
		self.bn2 = torch.nn.BatchNorm1d(h2_size)
		self.bn3 = torch.nn.BatchNorm1d(h3_size)

		self.output_layer = torch.nn.Linear(h3_size, 1)
		self.dropout = nn.Dropout(p)

	def forward(self, x):
		x = torch.nn.functional.relu(self.bn1(self.h1(x)))
		x = torch.nn.functional.relu(self.bn2(self.h2(x)))
		x = self.dropout(x)
		x = torch.nn.functional.relu(self.bn3(self.h3(x)))
		x = self.dropout(x)
		y = self.output_layer(x)
		return y

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


h1_sizes=[30,80,150,400,600]
h1_losses = []

lr = 1e-4
input_len = train_features.shape[1] 

n_epochs = 15

for h1 in h1_sizes:
	h1_size = h1
	h2_size = h1//2
	h3_size = h1//3

	model = NeuralNet(input_len, h1_size,h2_size, h3_size, 0.3)
	loss = nn.MSELoss(reduction="mean")
	optimizer = torch.optim.Adam(model.parameters(), lr=lr)

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

	h1_losses.append(val_losses)

plt.plot(h1_losses[0], label="h1 = 30")
plt.plot(h1_losses[1], label = "h1 = 80")
plt.plot(h1_losses[2], label= "h1 = 150")
plt.plot(h1_losses[3], label= "h1 = 400")
plt.plot(h1_losses[4], label= "h1 = 600")
plt.legend()
plt.savefig(snakemake.output[0])
plt.close()

wds=[0.001, 0.01, 0.1, 1, 10]
wd_losses = []

lr = 1e-4
input_len = train_features.shape[1] 
h1_size = 100
h2_size = 50
h3_size = 30
n_epochs = 15

for wd in wds:
	model = NeuralNet(input_len, h1_size,h2_size, h3_size, 0.3)
	loss = nn.MSELoss(reduction="mean")
	optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

	train_step = make_train_step(model, loss, optimizer)
	val_step = make_validation_step(model, loss)

	train_losses = []
	val_losses = []

	for epoch in range(n_epochs):
		#training
		train_loss = mini_batch_descent(train_loader, train_step)
		train_losses.append(train_loss)

		with torch.no_grad():
			val_loss = mini_batch_descent(val_loader, val_step)
			val_losses.append(val_loss)

	wd_losses.append(val_losses)

plt.plot(wd_losses[0], label="wd = 0.001")
plt.plot(wd_losses[1], label = "wd = 0.01")
plt.plot(wd_losses[2], label= "wd = 0.1")
plt.plot(wd_losses[3], label= "wd = 1")
plt.plot(wd_losses[4], label= "wd = 10")
plt.legend()
plt.savefig(snakemake.output[1])
plt.close()


lrs = [1e-3, 1e-4, 1e-5]
lr_losses = []

for lr in lrs:
	lr = lr
	input_len = train_features.shape[1] 
	h1_size = 25
	h2_size = 15
	h3_size = 8
	model = NeuralNet(input_len, h1_size,h2_size, h3_size, 0.5)
	loss = nn.MSELoss(reduction="mean")
	optimizer = torch.optim.Adam(model.parameters(), lr=lr)

	train_step = make_train_step(model, loss, optimizer)
	val_step = make_validation_step(model, loss)

	n_epochs = 20

	train_losses = []
	val_losses = []
	min_loss = 1

	for epoch in range(n_epochs):
		train_loss = mini_batch_descent(train_loader, train_step)
		train_losses.append(train_loss)

	lr_losses.append(train_losses)

plt.plot(lr_losses[0], label="lr = 1e-3")
plt.plot(lr_losses[1], label = "lr = 1e-4")
plt.plot(lr_losses[2], label= "lr = 1e-5")
plt.legend()
plt.savefig(snakemake.output[2])
plt.close()

batch_sizes = [1000,2000,4000,7000]
batch_size_losses = []


for batch_size in batch_sizes: 
	train_loader = DataLoader(dataset=train_data,batch_size=batch_size,shuffle=True)
	val_loader = DataLoader(dataset = val_data, batch_size=batch_size)

	lr = 1e-4
	input_len = train_features.shape[1] 
	h1_size = 100
	h2_size = 50
	h3_size = 30
	model = NeuralNet(input_len, h1_size,h2_size, h3_size, 0.3)
	loss = nn.MSELoss(reduction="mean")
	optimizer = torch.optim.Adam(model.parameters(), lr=lr)

	train_step = make_train_step(model, loss, optimizer)
	val_step = make_validation_step(model, loss)

	n_epochs = 15

	train_losses = []

	for epoch in range(n_epochs):
		train_loss = mini_batch_descent(train_loader, train_step)
		train_losses.append(train_loss)

	batch_size_losses.append(train_losses)

plt.plot(batch_size_losses[0], label="1000")
plt.plot(batch_size_losses[1], label = "2000")
plt.plot(batch_size_losses[2], label= "4000")
plt.plot(batch_size_losses[3], label= "7000")
plt.legend()
plt.savefig(snakemake.output[3])
plt.close()

lr = 1e-4
input_len = train_features.shape[1] 
h1_size = 100
h2_size = 50
h3_size = 30
n_epochs = 15

model = NeuralNet(input_len, h1_size,h2_size, h3_size, 0.3)
loss = nn.MSELoss(reduction="mean")
optimizer = torch.optim.SGD(model.parameters(), lr=lr)
train_step = make_train_step(model, loss, optimizer)

train_losses_SGD = []

for epoch in range(n_epochs):
	train_loss = mini_batch_descent(train_loader, train_step)
	train_losses_SGD.append(train_loss)

model = NeuralNet(input_len, h1_size,h2_size, h3_size, 0.3)
loss = nn.MSELoss(reduction="mean")
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
train_step = make_train_step(model, loss, optimizer)

train_losses_Adam = []

for epoch in range(n_epochs):
	train_loss = mini_batch_descent(train_loader, train_step)
	train_losses_Adam.append(train_loss)

model = NeuralNet(input_len, h1_size,h2_size, h3_size, 0.3)
loss = nn.MSELoss(reduction="mean")
optimizer = torch.optim.Adadelta(model.parameters(), lr=lr)
train_step = make_train_step(model, loss, optimizer)

train_losses_Adelta = []

for epoch in range(n_epochs):
	train_loss = mini_batch_descent(train_loader, train_step)
	train_losses_Adelta.append(train_loss)

plt.plot(train_losses_Adam, label="Adam")
plt.plot(train_losses_Adelta, label="Adelta")
plt.plot(train_losses_SGD, label="SGD")
plt.legend()
plt.savefig(snakemake.output[4])
plt.close()
