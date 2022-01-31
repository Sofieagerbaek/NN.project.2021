from typing import Sequence
import numpy as np
import math
import py2bit
import pyBigWig
import statistics

#Åbner 2bit-fil
ref = py2bit.open(snakemake.input.ref)
ref_chroms = ref.chroms()

#Åbner BigWig-filerne 
track = pyBigWig.open(snakemake.input.track)

window_size = snakemake.params.window
flank_size = window_size // 2

# The dictionary to one-hot encode
nt_to_one_of_j = {"A":"1\t0\t0\t0", "C":"0\t1\t0\t0", "G":"0\t0\t1\t0", "T":"0\t0\t0\t1"}

xchrx = snakemake.params.chroms

# To iterate over the chromosome
n = 0

for chr in xchrx:

	file = open(snakemake.output.result[n], "w") #creates a new file per chromosome
	chr_len = ref_chroms[chr] #længden fra reference genomet
		
	for pos in range(flank_size, chr_len-flank_size, window_size):

		seq = ref.sequence(chr, pos-flank_size, pos + flank_size+1)
   
		assert(len(seq) == window_size) #vores window passer med det anmodede

		if 'N' in seq:
			continue
		
		# Calculates the mean value over the window
		track_val = statistics.mean(track.values(chr, pos - flank_size, pos + flank_size + 1))

		if math.isnan(track_val):
			continue

		track_val = str(np.arcsinh(track_val))
		start = str(pos - flank_size)
		stop = str(pos + flank_size + 1)
		
		#one-hot encoding
		seq_one_of_j = []
		for i in range(len(seq)):
			seq_one_of_j.append(nt_to_one_of_j[seq[i]])

		seq = "\t".join(seq_one_of_j)
		
		#Writes a full row in the file for each window
		file.write(chr+"\t"+start+"\t"+stop+"\t"+track_val+"\t"+seq+"\n")

	file.close()
	n += 1

