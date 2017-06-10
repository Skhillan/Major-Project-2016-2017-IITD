from cosine_sim import cosine_sim
import heapq
import numpy as np
import os
import pickle

# flag = 0 takes array_of_indexes and flag = 1 takes words and vectors

def normalization(file_twitter, file_wiki, flag, array_of_indexes, theta,  words, vectors):

	ROOT = os.path.abspath(os.path.dirname(__file__))

	# finding wiki vocabulary
	f = open(ROOT + '/..' + file_wiki + '.txt', 'r')
	lines_1 = f.read().splitlines()
	f.close()

	f = open(ROOT + '/..' + file_twitter + '.txt', 'r')
	lines_2 = f.read().splitlines()
	f.close()


# wiki vocab
	vocab_wiki = []
	for line in lines_1:
		line.replace("\n", "")
		ar = line.split(' ', 1)
		word = ar[0]
		vector = np.array(map(float, ar[1].split(' ')))
		vocab_wiki.append([word, vector])


# # twitter vocab
# 	vocab_twitter = []
# 	for line in lines_2:
# 		line.replace("\n", "")
# 		ar = line.split(' ', 1)
# 		word = ar[0]
# 		vector = np.array(map(float, ar[1].split(' ')))
# 		vocab_twitter.append([word, vector])


# # pickle both vocabs

# 	pickle.dump( vocab_wiki, open( "vocab_wiki_all.p", "wb" ) )
# 	pickle.dump( vocab_twitter, open( "vocab_twitter_all.p", "wb"))

	if (flag == 0):
		words_normal = []
		for index in array_of_indexes:
			line = lines_2[index-1]
			line.replace("\n", "")
			ar = line.split(' ',1)
			x = np.array(map(float, ar[1].split(' ')))
			words_normal.append([ar[0], x])


		words_normal_vectors = []
		for x in words_normal:
			words_normal_vectors.append(x[1])

		o = len(words_normal_vectors)

		
		words_normal_vectors = np.c_[ np.ones(o), words_normal_vectors]
	else:
		words_normal = words
		words_normal_vectors = np.c_[ np.ones(o), vectors]


	words_transform = []
		# transform using transformation matrix first
	# print words_normal_vectors

	theta = [x[0] for x in theta]
	# print len(theta)
	# print len(theta[1])
	# loss = [x[1] for x in theta]


	for i in range(o):
		y_predict = []
		for t in theta:
			y_predict.append(np.dot(t, words_normal_vectors[i]))
		# print y_predict
		words_transform.append(y_predict)


	m = len(words_transform)
	# print m 
	# print len(words_transform[i])
	# print words_transform.shape


	closest_cluster = [] # top closest words
	top_sim = [] # top similarities array
	for i in range(m):
		top_n_sim, closest = find_closest(words_transform[i], vocab_wiki, 10, 'cosine')
		closest_cluster.append(closest)
		top_sim.append(top_n_sim)



	return words_transform, closest_cluster, words_normal, top_sim


def find_closest(word_vector, vocab, number, metric):
	vectors = np.array(word_vector)
	# print vectors
	# print len(vectors)
	top_number = -number
	similarity = []
	if (metric == 'cosine'):
		for c in vocab:
			sim = cosine_sim(c[1], vectors)
			similarity.append([c[0], sim])

	# print len(vocab)
	# print vocab[1]
	# print len(similarity)
	# print similarity[1]
	
	simi = [x[1] for x in similarity]

	top_n = np.argsort(simi)[-10:]
	# print top_n
	top_n_sim = []
	wiki_closest = []
	for j in top_n:
		top_n_sim.append(simi[j])
		wiki_closest.append(vocab[j])

	return top_n_sim, wiki_closest

