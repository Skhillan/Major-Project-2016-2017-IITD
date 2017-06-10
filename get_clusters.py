#get_clusters
#putting the clusters in arrays

from __future__ import print_function

import sys
import numpy as np
inFile = sys.argv[1]
outfile = sys.argv[2]



with open(inFile,'r') as i:
    lines = i.read().splitlines()

prefixes = []
words = []

for line in lines:
	l = line.split('\t')
	prefix = l[0]
	word = l[1]


	if prefix in prefixes:
		i = prefixes.index(prefix)
		words[i].append(word)
	else:
		prefixes.append(prefix)
		a = [word]
		words.append(a)

# c = 0
# for prefix in prefixes:
# 	print prefix+ '\t' + words[c] + '\n'
# 	c = c+1

  # Only needed for Python 2
f = open(outfile, 'w')



c = 0
for prefix in prefixes:
	f.write(prefix + "\t")
	f.write("\t".join(words[c]))
	f.write("\n")
	c = c+1


# sort clusters
