from typing import Sequence
import numpy
import argparse
import math
import py2bit
import pyBigWig

#Hvis vi bruger argsparse tager man argumenter i terminalen
#Så vidt jeg har forstået er den en måde at vi kan specificere
#input i workflow filen, så scriptet er mindre specifikt. (??)

parser = argparse.ArgumentParser(description="track.windows")

parser.add.argument('--track', dest = 'track', type = str)
parser.add.argument('--window_size' dest = 'window_size', type = int, default = 201)

args = parser.parse_args()

#Åbner BigWig filerne 
track = pyBigWig.open(args.track)

flank_size = args.window_size // 2

nr_pos = 0 #tæller for antallet af posisitioner løbende
nr_pos_skip_n = 0 #tæller hvor mange positioner der skippes
nr_pos_skip_na = 0 #tæller hvor mange skippes pga. na

#assert() kan bruges som tjek igennem koden, så hvis noget ikke 
#passer terminerer man før man bruger mere tid på det næste.

nt_to_one_of_j = {"A":"1\t0\t0\t0", "C":"0\t1\t0\t0", "G":"0\t0\t1\t0", "T":"0\t0\t0\t1"}


for chr in xchrx:

    chr_len = ref_chroms[chr] #længden fra reference genomet

    for pos in range(flank_size, chr_len - flank_size, args.window_size): #midter-pos i hver interval
        nr_pos += 1
        if nr_pos % 1e4 == 0: #tæller - forstår ikke helt koncept
            print(f"processed {nr_pos} positions")

        seq = ref.Sequence(chr, pos-flank_size, pos + flank_size+1) #definere vindue
        assert(len(seq) == args.window_size) #vores window passer med det anmodde

        if 'N' in seq: #position hvor man ikke har læst nucleotide
            nr_pos_skip_n += 1
            continue #Continue bryder loopet og går til næste vindue

        track_vals = track.values(chr, pos, pos+1) #.values er en måde at finde
        #average af values over en range i pyBigWig
        # - her tager vi kun én position af gangen i vores interval
        assert(len(track_vals) == 1) #test igen

        if math.isnan(track_vals[0]):
            nr_pos_skip_na += 1
            continue #tæller positioner med nan

        seq_one_of_j = [] #one-hot encoding
        for i in range(len(seq)):
            seq_one_of_j.append(nt_to_one_of_j[seq[i]])
        
        output .... .write()

output.close()



