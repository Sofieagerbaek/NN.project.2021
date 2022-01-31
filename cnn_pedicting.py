import numpy as np
import torch
import pandas as pd
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from scipy.stats import stats

chr12_df = pd.read_csv(snakemake.input[0], sep="\t")
test_features = torch.load(snakemake.input[1])
test_labels = torch.as_tensor(chr12_df.iloc[:,3].values)

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
input_size = 804
num_channels = 32
h1_size = 150
pool_kernel_size = 2
conv_kernel_size = 48
conv_stride = 4

features = torch.unsqueeze(test_features, 1)

model = conv_para_network(input_size, num_channels, conv_kernel_size, conv_stride, pool_kernel_size, h1_size)

features=features.type(torch.LongTensor)

model.load_state_dict(torch.load(snakemake.input[1]))
model.eval()

with torch.no_grad():
	y_pred = model(features)


#Making a bed file for visualization and checking squared error against sequence

y_pred = y_pred.numpy()

seq = test_features.numpy()

seq_ny = []
for i in range(len(seq)):
	lst = []
	for j in range(0,len(seq[0]),4):
		lst.append(''.join(str(e) for e in seq[i][j:j+4].astype('int')))
	seq_ny.append(lst)

print(seq_ny[0])

onehot_dict = {"1000":"A", "0100":"C", "0010":"G", "0001":"T"}

seq_win = []
for i in range(len(seq_ny)):
  seqe = ''
  for j in range(201):
  	if seq_ny[i][j] == '':
  		continue
  	seqe+=onehot_dict[seq_ny[i][j]]
  seq_win.append(seqe)

print(seq_win[0])

visu_bed = pd.DataFrame()
visu_bed['chromosome'] = chr12_df.iloc[:,0]
visu_bed['start_pos'] = chr12_df.iloc[:,1]
visu_bed['end_pos'] = chr12_df.iloc[:,2]
visu_bed['sequence'] = seq_win
visu_bed['y_true'] = chr12_df.iloc[:,3]
visu_bed['y_pred'] = y_pred


#For correlation between sequence GC content and squared error

squared_error = (visu_bed['y_true']-visu_bed['y_pred'])**2

visu_bed['squared_error'] = squared_error

visu_bed.to_csv(snakemake.output[0], sep="\t", header=False, index=False)

def GC_content(x):
	GC = 0
	for i in x:
		if i=='G' or i=='C': 
			GC += 1
		continue
	GC_content = GC/len(x)
	return GC_content

print(visu_bed['sequence'][0])

GC_cont = []
for i in range(len(visu_bed['sequence'])):
	GC_cont.append(GC_content(visu_bed['sequence'][i]))


print(stats.pearsonr(squared_error, GC_cont))

plt.scatter(squared_error, GC_cont)
plt.legend(stats.pearsonr(squared_error, GC_cont), loc="upper right")
plt.xlabel("Squared error")
plt.ylabel("GC content")
plt.savefig(snakemake.output[1])
plt.close()