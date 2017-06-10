import os
from multiprocessing import Manager
from sklearn.decomposition import PCA
import pylab as Plot
from sklearn.linear_model import SGDClassifier
import gradient_descent_2
import dim_reduce_plot
import pickle
import plot_generator
import numpy as np


def pre_process(file_name_twitter, file_name_wiki, number):

    ROOT = os.path.abspath(os.path.dirname(__file__))

    manager = Manager()
    global v_common, v_wiki, v_twitter, v_indices, c_twitter, c_wiki, word, v_twit, v_wik
    global vec_pairs, twitter_vectors, wiki_vectors, twitter_matrix, wiki_matrix

    f = open(ROOT + '/..' + file_name_twitter + '.txt', 'r')
    lines_1 = f.read().splitlines()
    f.close()

    f = open(ROOT + '/..' + file_name_wiki + '.txt', 'r')
    lines_2 = f.read().splitlines()
    f.close()

    n_words = number

    # twitter vocabulary
    v_twitter = []
    for line in lines_1[:n_words]:
        word = line.split(' ', 1)[0]
        v_twitter.append(word)

    # wiki vocabulary
    v_wiki = []
    for line in lines_2[:n_words]:
        word = line.split(' ', 1)[0]
        v_wiki.append(word)

    # print len(v_twitter)
    # print len(v_wiki)

    # ease of computation

    v_twit = v_twitter[:]
    v_wik = v_wiki[:]

    # parallel processing on this list
    # c_wiki = []
    # for i in range(len(v_wik)):
    #     c_wiki.append(i)

    # extra vocabulary for twiter
    twitter_extra_w = []
    twitter_extra_vec = []
    extra_indices = []


    # common vocabulary
    v_common = []
    v_indices = []
    c_twitter = 0
    for word in v_twit:
        flg = 0
        c_wiki = 0
        for word1 in v_wik:
            if(word == word1):
                flg = 1
                v_common.append(word)
                v_indices.append([c_twitter, c_wiki])
                break

            c_wiki +=1

        if flg == 0:
            twitter_extra_w.append(word)
            extra_indices.append(c_twitter)            
        #         v_indices.append([c_twitter, n])
        # p = multiprocessing.Pool(2)
        # p.map(find_wiki,c_wiki)
        c_twitter +=1

    print len(v_indices)
    # vector pairs for common words
    vec_pairs = []
    twitter_vectors = []
    wiki_vectors = []
    for index in v_indices:
        t_index = index[0]
        w_index = index[1]
        t_line = lines_1[t_index].replace("\n", "")
        w_line = lines_2[w_index].replace("\n", "")
        t_vec = t_line.split(' ', 1)[1]
        w_vec = w_line.split(' ', 1)[1]
        x = np.array(map(float, t_vec.split(' ')))
        y = np.array(map(float, w_vec.split(' ')))
        twitter_vectors.append(x)
        wiki_vectors.append(y)
        # vec_pairs.append([x,y])

# ----------------------------------------- uncomment for extra vectors to test from twitter ---------------------------------------

    ## extra vectors from twitter

    # for index in extra_indices:

    #     line = lines_1[index].replace("\n", "")
    #     vec = line.split(' ', 1)[1]
    #     v = np.array(map(float, vec.split(' ')))
    #     twitter_extra_vec.append(v)

    # for i in range(10):
    #     print twitter_extra_w[i]

    # file1 = open('extra_vecs_' + str(n_words)+ 'twitter.p', 'wb') # vectors of extra words
    # pickle.dump(twitter_extra_vec, file1)
    # file1.close()

    # file2 = open('extra_words_' + str(n_words)+ 'twitter.p', 'wb') # extra words
    # pickle.dump(twitter_extra_w, file2)
    # file2.close()

    # print len(vec_pairs[:])
    # print len(twitter_vectors)



    # print cosine_sim(vec1, vec2)

    # visualizing word vectors using PCA

    # dim_reduce_plot.dim_reduce_plot(twitter_vectors, wiki_vectors, v_common)

    # # Visualizing dimensionality reduction


    # transform Twitter to Wiki vectors

    # twitter to every component of wiki vector

#-------------------------------------- uncomment for pre processing -----------------------------------

    common_pairs = len(v_common)

    ind = int(common_pairs*0.8) # index for training and testing

    t_vectors = np.array(wiki_vectors)
    d = t_vectors.shape[1]
    var_train = []
    var_test = []
    var = []
    for j in range(0, d):
        x = []
        y = []
        z = []
        k = 0
        for vector in wiki_vectors:
            if k < ind:
                x.append(vector[j])
                z.append(vector[j])
                k = k+1
            else:
                y.append(vector[j])
                z.append(vector[j])
                k = k+1
        var_train.append(np.array(x))
        var_test.append(np.array(y))
        var.append(np.array(z))
            
    

    # d target variables

    
    # # print tv1

    X = np.array(twitter_vectors)

    print len(X)    
    return X, v_common, var_train, var_test, ind, d, var


# ---------------------------------- uncomment for pre-processing ---------------------------------
    
     # insert column



    # testing and training

    

    # file1 = open('predicted_vecs_' + str(n_words)+ '_common.p', 'wb')
    # pickle.dump(predicted_vecs, file1)
    # file1.close()

    # file2 = open('wiki_vecs_' + str(n_words)+ '_common.p', 'wb')
    # pickle.dump(wiki_vectors, file2)
    # file2.close()

    # file3 = open('twitter_vecs_' + str(n_words)+ '_common.p', 'wb')
    # pickle.dump(twitter_vectors, file3)
    # file3.close()

    # file4 = open('common_vocab_' + str(n_words)+ '_common.p', 'wb')
    # pickle.dump(v_common, file4)
    # file4.close()



#     # wiki_vectors = vec_pairs[:][1]
#     #
#     # target_matrix = np.matrix(twitter_vectors[:10])
#     # print target_matrix
#     # print X[1]
#     # pca = PCA(n_components=2)
#     # pca.fit(twitter_vectors)
#     # pca.fit_transform(twitter_vectors)
#     # print(pca.explained_variance_ratio_)
#     # print twitter_vectors

#     # pca = PCA(n_components=10)
#     # Y = pca.fit(X).transform(X)
#     # print Y

#     # plt.scatter(Y[:, 0], Y[:, 1])
#     # for label, x, y in zip(v_common[:10], Y[:, 0], Y[:, 1]):
#     #     plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points')
#     # plt.show()

#     # print twitter_vectors[1]
#     # print wiki_vectors[1]



# # def find_wiki(n):
# #     if (word == v_wik[n]):
# #         v_common.append(word)
# #         v_indices.append([c_twitter, n])
# #         return

# ----------- uncomment for individual run -------------------

# twitter_file = '/twitter/glove.twitter.27B.50d'
# wiki_file = '/wiki/glove.6B.50d'
# number_of_words = 30000

# pre_process(twitter_file, wiki_file, number_of_words)