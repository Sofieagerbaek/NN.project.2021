import numpy as np
import torch
from sklearn.model_selection import train_test_split as split
import torch.nn as nn
from torch.utils.data import Dataset, TensorDataset, DataLoader
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot as plt
import copy

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

#Initializing
lr = 1e-4
input_len = train_features.shape[1] 
h1_size = 150
h2_size = 70
h3_size = 35
n_epochs = 8

train_loss_file = open(snakemake.output[0], 'w')
val_loss_file = open(snakemake.output[1], 'w')

smallest_loss = 1

for i in range(50):
	model = NeuralNet(input_len, h1_size,h2_size, h3_size, 0.3)
	loss = nn.MSELoss(reduction="mean")
	optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay = 0.01)

	train_step = make_train_step(model, loss, optimizer)
	val_step = make_validation_step(model, loss)
	min_loss = 1
	
	for epoch in range(n_epochs):
		train_loss = mini_batch_descent(train_loader, train_step)

		with torch.no_grad():
			val_loss = mini_batch_descent(val_loader, val_step)
			if val_loss < min_loss:
				min_loss = val_loss
				small_loss_model = copy.deepcopy(model.state_dict())

	val_loss_file.write(str(val_loss)+' ')
	train_loss_file.write(str(train_loss)+' ')

	if min_loss < smallest_loss: 
		smallest_loss = min_loss
		smallest_loss_model = small_loss_model


torch.save(small_loss_model, snakemake.output[2])

train_loss_file.close()
val_loss_file.close()