import os
import matplotlib.pyplot as plt


def pre_process(file_name_twitter, file_name_wiki):
    ROOT = os.path.abspath(os.path.dirname(__file__))

    f = open(ROOT + '/Documents/Sem 9@iitd/MTP' + file_name_twitter + '.txt', 'r')
    lines = f.read().splitlines()
    f.close()

    print lines[1]
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


pre_process('/twitter/glove.twitter.27B.25d.txt','/wiki/glove.6B.50d.txt')