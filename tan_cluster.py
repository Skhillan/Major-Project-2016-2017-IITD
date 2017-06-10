# tan clusters
# give the file containing tan clusters

import sys
inFile = sys.argv[1]
# outFile = sys.argv[2]

with open(inFile,'r') as i:
    lines = i.splitlines()


prefixes = []
words = []

for line in lines:
	words = line.split('\t')
	words.append(words[0])
	prefixes.append(words[1])


print (words[x] for x in range(10))
print (prefixes[x] for x in range(10))