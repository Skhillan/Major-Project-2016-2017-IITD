import pickle

file8 = open('extra_vecs_30000twitter.p', 'rb')
vectors= pickle.load(file8)
file8.close()

print "Reached here"

file00 = open('extra_vecs_30000twitter.p', 'rb')
words= pickle.load(file00)
file00.close()

print len(words)
print words [1]
print len(vectors)
print vectors[1]