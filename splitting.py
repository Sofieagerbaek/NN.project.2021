#Splitting 2 tsv files into 6 - train/val/test
import numpy as np
from sklearn.model_selection import train_test_split

seq = open(snakemake.input[0], "R")
val = open(snakemake.input[1], "R")

seq_train_split = open(snakemake.output[0], "W")
seq_val_split = open(snakemake.output[1], "W")
seq_test_split = open(snakemake.output[2], "W")

val_train_split = open(snakemake.output[3], "W")
val_val_split = open(snakemake.output[4], "W")
val_test_split = open(snakemake.output[5], "W")

#possibly do an assertion to check that seq and val have the same dimensions

seq_train, seq_val.test, val_train, val_val.test = train_test_split(seq, val, test.size = 0.1)

