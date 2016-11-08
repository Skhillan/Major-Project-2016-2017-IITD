import os
import numpy as np
# import multiprocessing
from multiprocessing import Manager

# cosine similarity


def cosine_sim(vec1, vec2):
    if(len(vec1)==len(vec2)):
        dot_value = np.dot(vec1, vec2)
        vec1_mod = np.sqrt((vec1 * vec1).sum())
        vec2_mod = np.sqrt((vec2 * vec2).sum())
        cos_angle = dot_value / vec1_mod / vec2_mod  # cosine of angle between x and y
        return cos_angle
    else:
        print "Vector dimensions don't match"
        return


def pre_process(file_name_twitter, file_name_wiki):
    ROOT = os.path.abspath(os.path.dirname(__file__))

    manager = Manager()
    global v_common, v_wiki, v_twitter, v_indices, c_twitter, c_wiki, word, v_twit, v_wik
    global vec_pairs

    f = open(ROOT + '/MTP' + file_name_twitter + '.txt', 'r')
    lines_1 = f.read().splitlines()
    f.close()

    f = open(ROOT + '/MTP' + file_name_wiki + '.txt', 'r')
    lines_2 = f.read().splitlines()
    f.close()

    # twitter vocabulary
    v_twitter = []
    for line in lines_1[:1000]:
        word = line.split(' ', 1)[0]
        v_twitter.append(word)

    # wiki vocabulary
    v_wiki = []
    for line in lines_2[:1000]:
        word = line.split(' ', 1)[0]
        v_wiki.append(word)

    # print len(v_twitter)
    # print len(v_wiki)

    # ease of computation

    v_twit = v_twitter[:100]
    v_wik = v_wiki[:100]

    # parallel processing on this list
    # c_wiki = []
    # for i in range(len(v_wik)):
    #     c_wiki.append(i)

    # common vocabulary
    v_common = manager.list()
    v_indices = []
    c_twitter = 0
    for word in v_twit:
        c_wiki = 0
        for word1 in v_wik:
            if(word == word1):
                v_common.append(word)
                v_indices.append([c_twitter, c_wiki])
                break
            c_wiki +=1
        #         v_indices.append([c_twitter, n])
        # p = multiprocessing.Pool(2)
        # p.map(find_wiki,c_wiki)
        c_twitter +=1

    # vector pairs for common words
    vec_pairs = []
    for index in v_indices:
        t_index = index[0]
        w_index = index[1]
        t_line = lines_1[t_index].replace("\n", "")
        w_line = lines_2[w_index].replace("\n", "")
        t_vec = t_line.split(' ', 1)[1]
        w_vec = w_line.split(' ', 1)[1]
        x = np.array(t_vec.split(' '), dtype = float)
        y = np.array(w_vec.split(' '), dtype = float)
        vec_pairs.append([x,y])

    # print len(v_common)
    # print v_common[7]
    # print v_indices[7]
    # print "this is the wiki vector\n"
    # print len(vec_pairs[7][0])
    # print len(vec_pairs[7][1])
    # print vec_pairs[7][1]

    vec1 = vec_pairs[7][0]
    vec2 = vec_pairs[7][1]


    print cosine_sim(vec1, vec2)



    # features = []  # list of all the features

    # if not os.path.exists(ROOT + '/filtered_data'):
    #     os.makedirs(ROOT + '/filtered_data')
    #
    # # remove data points with label 2/1.00
    # f1 = open(ROOT + '/filtered_data/filtered_label_' + file_name + '.txt', 'w')
    # flag = 0
    # for line in lines:
    #     p = str.split(line, '|')
    #     f_name = p[0]
    #     if f_name == '2':
    #         flag = 1
    #     if f_name == '':
    #         flag = 0
    #     if not (flag == 1):
    #         f1.write(line)
    #         f1.write('\n')
    # f1.close()
    #
    # f = open(ROOT + '/filtered_data/filtered_label_' + file_name + '.txt', 'r')
    # lines = f.read().splitlines()
    # f.close()
    #
    # for line in lines:
    #     p = str.split(line, '|')
    #     f_name = p[0]
    #     if len(f_name) > 2:
    #         features.append(f_name)
    #
    # unique_features = Counter(features)  # list of all the unique features
    # count = 0.0
    # other = 0.0
    # for feature in unique_features:
    #     if unique_features[feature] == 1:
    #         count += 1
    #     else:
    #         other += 1
    #
    # # # Uncomment the following part for single file histogram
    # # y = []
    # # for x in unique_features:
    # #     y.append(unique_features[x])
    # #
    # # plt.hist(y, bins=1000, histtype='step')
    # # plt.show()
    #
    # f1 = open(ROOT + '/filtered_data/filtered_feature_' + file_name + '.txt',
    #           'w')  # remove features which appear only once
    # for line in lines:
    #     p = str.split(line, '|')
    #     f_name = p[0]
    #     if len(f_name) < 3:  # check if its not a feature
    #         f1.write(line)
    #         f1.write('\n')
    #     else:
    #         if not unique_features[f_name] == 1:  # check if the feature appears only once
    #             f1.write(line)
    #             f1.write('\n')
    # f1.close()
    # return features

# def find_wiki(n):
#     if (word == v_wik[n]):
#         v_common.append(word)
#         v_indices.append([c_twitter, n])
#         return
pre_process('/twitter/glove.twitter.27B.50d','/wiki/glove.6B.50d')