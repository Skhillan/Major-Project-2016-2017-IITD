## searching for a word in the cluster to be a word in wiki vocabulary

# make wiki vocab pickle

# remember previous algo

# for a word in twitter, find the cluster, search for all words in the cluster in wiki vocab -> yes/no

# main with brown clustering

from pre_process import pre_process
from plot_generator import plot_generator
from gradient_descent_2 import gradient_descent_2
from gradient_descent_2 import main_function
from gradient_descent_2 import main_function_with_plot
from gradient_descent_2 import normalize
from dim_reduce_plot import dim_reduce_plot
import numpy as np
from normalization import normalization
import pickle
import os

twitter_file = '/twitter/glove.twitter.27B.50d'
wiki_file = '/wiki/glove.6B.50d'
number_of_words = 30000
learning_rate = 0.03
iterations = 2000
start = 10
end = 20
array_of_indexes = [1154, 1155, 1224, 1244, 1306, 1380, 1471, 1567, 11778, 3659, 3600, 3596, 3576, 3524, 3516, 3517, 3443, 3422]


# X, v_common, var_train, var_test, ind, d, var = pre_process(twitter_file, wiki_file, number_of_words)

# v_common_test = v_common[ind:]
# X_train = X[:ind]
# X_test = X[ind:]

# # ----------------------------------------------------------
# # to make loss plot for each variable

# # predicted_vecs = []
# #     for i in range(0,d):
# #         title = 'variable_no._'+str(i)
# #         y_predict = gradient_descent_2.main_function_with_plot(X_train,var_train[i],X_test, var_test[i], learning_rate, iterations, title)
# #         predicted_vecs.append(y_predict)

# # -----------------------------------------------------------
# ## To look at test and train data plots

# # predicted_vecs = []
# # for i in range(0,d):
# #     y_predict = main_function(X_train,var_train[i],X_test, var_test[i], learning_rate, iterations)
# #     predicted_vecs.append(y_predict)

# # predicted_vecs = np.array(predicted_vecs).transpose()

# # ------------------------------------------------------------
# ## for normalization

# theta_loss = []
# for i in range(0,d):
# 	theta, J = normalize(X, var[i], learning_rate, iterations)
# 	theta_loss.append([theta, J])

# file4 = open('theta_loss.p', 'wb')
# pickle.dump(theta_loss, file4)
# file4.close()


theta_losses = pickle.load(open('theta_loss.p', 'rb'))
words_transform, closest_cluster, words_normal = normalization(twitter_file, wiki_file, array_of_indexes, theta_losses)
print "this is done, got the clusters"

arrays = []
for c in closest_cluster:
	ar = []
	for i in c:
		ar.append(i[0])
	arrays.append(ar)

all_words = []
for w in words_normal:
	all_words.append(w[0])

print arrays
print all_words
# plot_generator(len(X_test), start, end, predicted_vecs, np.array(var_test).transpose(), X_test, v_common_test)



# dissimilarity
# string edit distance
# MDS
# parallel processing
# Latent semantic analysis
# 


# make arrays of clusters for search

ROOT = os.path.abspath(os.path.dirname(__file__))
inFile = open(ROOT + '/..' + '52m_tweets_clusters' + '.txt', 'r')


lines = inFile.read().splitlines()

prefixes = []
words = []

for line in lines:
	l = line.split('\t')
	prefix = l[0]

	if prefix in prefixes:
		i = prefixes.index(prefix)
		words[i] = words[i].append(l[1])

	else:
		prefixes.append(prefix)
		words.append([l[1]])

c = 0
for prefix in prefixes:
	print prefix+ '\t' + words[c] + '\n'
	c = c+1