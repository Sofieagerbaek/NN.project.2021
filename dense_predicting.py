import numpy as np
import torch
import pandas as pd
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from scipy.stats import stats

test_features = torch.load(snakemake.input[0])
chr12_df = pd.read_csv(snakemake.input[1], sep="\t")

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

input_len = 804
h1_size = 150
h2_size = 70
h3_size = 35

model = NeuralNet(input_len, h1_size,h2_size, h3_size, 0.3)

model.load_state_dict(torch.load(snakemake.input[2]))
model.eval()

with torch.no_grad():
	y_pred = model(test_features)

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

squared_error = (visu_bed["y_true"]-visu_bed["y_pred"])**2

visu_bed['squared_error'] = squared_error

visu_bed.iloc[1:].to_csv(snakemake.output[0], sep="\t", header=False, index=False)

def GC_content(x):
	GC = 0
	for i in x:
		if i=='G' or i=='C': 
			GC += 1
		continue
	GC_content = GC/len(x)
	return GC_content

GC_cont = []
for i in range(len(visu_bed['sequence'])):
	GC_cont.append(GC_content(visu_bed['sequence'][i]))

plt.scatter(squared_error, GC_cont)
plt.legend(stats.pearsonr(squared_error, GC_cont), loc="upper right")
plt.xlabel("Squared error")
plt.ylabel("GC content")
plt.savefig(snakemake.output[0])
plt.close()
