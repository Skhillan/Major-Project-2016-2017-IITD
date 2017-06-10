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

# file4 = open('theta_loss.p', 'wb')
# pickle.dump(theta_loss, file4)
# file4.close()


twitter_file = '/twitter/glove.twitter.27B.50d'
wiki_file = '/wiki/glove.6B.50d'
number_of_words = 30000
learning_rate = 0.03
iterations = 2000
start = 10
end = 20
f = 1
# array_of_indexes = [1154, 1155, 1224, 1244, 1306, 1380, 1471, 1567, 11778, 3659, 3600, 3596, 3576, 3524, 3516, 3517, 3443, 3422]


print "started here"
#### --------------------------------------- Uncomment for any of the below options ----------------------------------------------------


# X, v_common, var_train, var_test, ind, d, var = pre_process(twitter_file, wiki_file, number_of_words)

# v_common_test = v_common[ind:]
# X_train = X[:ind]
# X_test = X[ind:]

#### ---------------------------------------- Uncomment to make loss plot for each variable ---------------------------------------------

# # predicted_vecs = []
# #     for i in range(0,d):
# #         title = 'variable_no._' + str(i)
# #         y_predict = gradient_descent_2.main_function_with_plot(X_train,var_train[i],X_test, var_test[i], learning_rate, iterations, title)
# #         predicted_vecs.append(y_predict)

#### -------------------------------------------Uncomment to look at test and train data plots -------------------------------------------

# # predicted_vecs = []
# # for i in range(0,d):
# #     y_predict = main_function(X_train,var_train[i],X_test, var_test[i], learning_rate, iterations)
# #     predicted_vecs.append(y_predict)

# # predicted_vecs = np.array(predicted_vecs).transpose()

#### --------------------------------------------Uncomment for getting theta losses ------------------------------------------------------

# theta_loss = []
# for i in range(0,d):
# 	theta, J = normalize(X, var[i], learning_rate, iterations)
# 	theta_loss.append([theta, J])

# file4 = open('theta_loss.p', 'wb')
# pickle.dump(theta_loss, file4)
# file4.close()

#### ------------------------------------------- uncomment for clusters etc. --------------------------------------

file8 = open('extra_vecs_30000twitter.p', 'rb')
vectors= pickle.load(file8)
file8.close()

vectors = vectors[:25]

print "Reached here"

file00 = open('extra_vecs_30000twitter.p', 'rb')
words= pickle.load(file00)
file00.close()

words = words[:25]

print len(words)
print words [1]
print len(vectors)
print vectors[1]
file9 = open('theta_loss.p', 'rb')
theta_losses = pickle.load(file9)
file9.close()

if (f == 0):
	words_transform, closest_cluster, words_normal, similarity = normalization(twitter_file, wiki_file, f , array_of_indexes, theta_losses, [], [])
else:
	print "it was here"
	words_transform, closest_cluster, words_normal, similarity = normalization(twitter_file, wiki_file, f , [], theta_losses, words, vectors)



arrays = []
for c in closest_cluster:
	ar = []
	for i in c:
		ar.append(i[0])
	arrays.append(ar)

l = len(arrays)

for k in range(l):
	s = len(arrays[k])
	print words_normal[k],
	print '	',
	print arrays[k],
	print similarity[k]



# Find no. of word vectors with similarity > 0.75

# count = 0
# for t in range(l):
# 	if (int(arrays[t][-1]) > 0.75):
# 		count = count + 1




# print arrays
# # print words_normal

# print words_normal
# print similarity

## plot_generator(len(X_test), start, end, predicted_vecs, np.array(var_test).transpose(), X_test, v_common_test)



# dissimilarity
# string edit distance
# MDS
# parallel processing
# Latent semantic analysis
# 