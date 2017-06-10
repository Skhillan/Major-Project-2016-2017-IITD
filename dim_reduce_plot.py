import pylab as Plot
from sklearn.manifold import TSNE  
import numpy as np



def dim_reduce_plot(vecs_twitter, vecs_wiki, words, title, subplot_no, start, end):
    
    X = vecs_twitter
    Y = vecs_wiki
    v_common = words

    tsne = TSNE(n_components = 2, random_state = 0)
    np.set_printoptions(suppress=True)
    reduced_matrix_twitter = tsne.fit_transform(X)

    tsne = TSNE(n_components = 2, random_state = 0)
    np.set_printoptions(suppress=True)
    reduced_matrix_wiki = tsne.fit_transform(Y)


    # Visualizing dimensionality reduction

    
    ax = Plot.subplot(1,2,subplot_no)
    max_x_twitter = np.amax(reduced_matrix_twitter, axis=0)[0]
    max_y_twitter = np.amax(reduced_matrix_twitter, axis=0)[1]
    max_x_wiki = np.amax(reduced_matrix_wiki, axis=0)[0]
    max_y_wiki = np.amax(reduced_matrix_wiki, axis=0)[1]
    max_x = max(max_x_twitter, max_x_wiki)
    max_y = max(max_y_twitter, max_y_wiki)
    # print max_x_twitter, max_x_wiki, max_x
    # print max_y_twitter, max_y_wiki, max_y

    Plot.xlim((-200, 200))
    # Plot.ylim((-max_y, max_y))
    Plot.ylim((-600, 600))


    # Plot all words

    # ax.scatter(reduced_matrix_twitter[:, 0], reduced_matrix_twitter[:, 1], color="red", marker = "s");
    # ax.scatter(reduced_matrix_wiki[:, 0], reduced_matrix_wiki[:, 1], color="blue", marker = "o");


    # for i in range(0, len(X)):
    #     target_word = v_common[i]
    #     print target_word
    #     x = reduced_matrix_twitter[i, 0]
    #     y = reduced_matrix_twitter[i, 1]
    #     x1 = reduced_matrix_wiki[i, 0]
    #     y1 = reduced_matrix_wiki[i, 1]        
    #     ax.annotate(target_word, (x, y))
    #     ax.annotate(target_word, (x1, y1))

    # Plot only  few words
    ax.scatter(reduced_matrix_twitter[start:end, 0], reduced_matrix_twitter[start:end, 1], color="red", marker = "s", label ='twitter_vectors');
    ax.scatter(reduced_matrix_wiki[start:end, 0], reduced_matrix_wiki[start:end, 1], color="blue", marker = "o", label = 'wiki_vectors');
    Plot.title(title + ' vectors')
    ax.legend()

    for i in range(start, end):
        target_word = v_common[i]
        # print target_word
        x = reduced_matrix_twitter[i, 0]
        y = reduced_matrix_twitter[i, 1]
        x1 = reduced_matrix_wiki[i, 0]
        y1 = reduced_matrix_wiki[i, 1]        
        ax.annotate(target_word, (x, y))
        ax.annotate(target_word, (x1, y1))

    # Plot.show()
    # Plot.savefig(title +'_word_vectors.png');

def corrected_plot(vecs_twitter, vecs_wiki, vecs_predicted, words, title, subplot_no, start, end):
    
    X = vecs_twitter
    Y = vecs_wiki
    v_common = words
    Z = vecs_predicted

    tsne = TSNE(n_components = 2, random_state = 0)
    np.set_printoptions(suppress=True)
    reduced_matrix_twitter = tsne.fit_transform(X)

    tsne = TSNE(n_components = 2, random_state = 0)
    np.set_printoptions(suppress=True)
    reduced_matrix_wiki = tsne.fit_transform(Y)

    tsne = TSNE(n_components = 2, random_state = 0)
    np.set_printoptions(suppress=True)
    reduced_matrix_predicted = tsne.fit_transform(Z)



    corrected_vectors = []

    c = 0
    for v in reduced_matrix_wiki:
        p = reduced_matrix_predicted[c]
        t = reduced_matrix_twitter[c]
        if (dist(p,v) < dist(t,v)):
            corrected_vectors.append(p)
        else:
            corrected_vectors.append(t)
        c = c+1

    if not(len(corrected_vectors) == len(reduced_matrix_predicted)):
        print ("warning..........")

    # Visualizing dimensionality reduction

    
    ax = Plot.subplot(1,2,subplot_no)
    max_x_twitter = np.amax(corrected_vectors, axis=0)[0]
    max_y_twitter = np.amax(corrected_vectors, axis=0)[1]
    max_x_wiki = np.amax(reduced_matrix_wiki, axis=0)[0]
    max_y_wiki = np.amax(reduced_matrix_wiki, axis=0)[1]
    max_x = max(max_x_twitter, max_x_wiki)
    max_y = max(max_y_twitter, max_y_wiki)
    # print max_x_twitter, max_x_wiki, max_x
    # print max_y_twitter, max_y_wiki, max_y

    Plot.xlim((-200, 200))
    # Plot.ylim((-max_y, max_y))
    Plot.ylim((-600, 600))

    # Plot all words

    # ax.scatter(reduced_matrix_twitter[:, 0], reduced_matrix_twitter[:, 1], color="red", marker = "s");
    # ax.scatter(reduced_matrix_wiki[:, 0], reduced_matrix_wiki[:, 1], color="blue", marker = "o");


    # for i in range(0, len(X)):
    #     target_word = v_common[i]
    #     print target_word
    #     x = reduced_matrix_twitter[i, 0]
    #     y = reduced_matrix_twitter[i, 1]
    #     x1 = reduced_matrix_wiki[i, 0]
    #     y1 = reduced_matrix_wiki[i, 1]        
    #     ax.annotate(target_word, (x, y))
    #     ax.annotate(target_word, (x1, y1))

    print corrected_vectors
    # Plot only  few words
    ax.scatter(np.array(corrected_vectors)[start:end, 0], np.array(corrected_vectors)[start:end, 1], color="red", marker = "s", label ='twitter_vectors');
    ax.scatter(reduced_matrix_wiki[start:end, 0], reduced_matrix_wiki[start:end, 1], color="blue", marker = "o", label = 'wiki_vectors');
    Plot.title(title + ' vectors')
    ax.legend()

    for i in range(start, end):
        target_word = v_common[i]
        # print target_word
        x = np.array(corrected_vectors)[i, 0]
        y = np.array(corrected_vectors)[i, 1]
        x1 = reduced_matrix_wiki[i, 0]
        y1 = reduced_matrix_wiki[i, 1]      
        ax.annotate(target_word, (x, y))
        ax.annotate(target_word, (x1, y1))

def dim_reduce_plot_normalized(vecs_twitter, vecs_wiki, words, title, subplot_no):
    
    X = vecs_twitter
    Y = vecs_wiki
    v_common = words

    tsne = TSNE(n_components = 2, random_state = 0)
    np.set_printoptions(suppress=True)
    reduced_matrix_twitter = tsne.fit_transform(X)

    tsne = TSNE(n_components = 2, random_state = 0)
    np.set_printoptions(suppress=True)
    reduced_matrix_wiki = tsne.fit_transform(Y)


    # Visualizing dimensionality reduction

    
    ax = Plot.subplot(1,2,subplot_no)
    max_x_twitter = np.amax(reduced_matrix_twitter, axis=0)[0]
    max_y_twitter = np.amax(reduced_matrix_twitter, axis=0)[1]
    max_x_wiki = np.amax(reduced_matrix_wiki, axis=0)[0]
    max_y_wiki = np.amax(reduced_matrix_wiki, axis=0)[1]
    max_x = max(max_x_twitter, max_x_wiki)
    max_y = max(max_y_twitter, max_y_wiki)
    # print max_x_twitter, max_x_wiki, max_x
    # print max_y_twitter, max_y_wiki, max_y

    Plot.xlim((-max_x, max_x))
    # Plot.xlim((-200, 200))
    Plot.ylim((-max_y, max_y))
    # Plot.ylim((-600, 600))


    # Plot all words

    # ax.scatter(reduced_matrix_twitter[:, 0], reduced_matrix_twitter[:, 1], color="red", marker = "s");
    # ax.scatter(reduced_matrix_wiki[:, 0], reduced_matrix_wiki[:, 1], color="blue", marker = "o");


    # for i in range(0, len(X)):
    #     target_word = v_common[i]
    #     print target_word
    #     x = reduced_matrix_twitter[i, 0]
    #     y = reduced_matrix_twitter[i, 1]
    #     x1 = reduced_matrix_wiki[i, 0]
    #     y1 = reduced_matrix_wiki[i, 1]        
    #     ax.annotate(target_word, (x, y))
    #     ax.annotate(target_word, (x1, y1))

    # Plot only  few words
    ax.scatter(reduced_matrix_twitter[:, 0], reduced_matrix_twitter[:, 1], color="red", marker = "s", label ='twitter_vectors');
    ax.scatter(reduced_matrix_wiki[:, 0], reduced_matrix_wiki[:, 1], color="blue", marker = "o", label = 'wiki_vectors');
    Plot.title(title + ' vectors')
    ax.legend()

    for i in range(start, end):
        target_word = v_common[i]
        # print target_word
        x = reduced_matrix_twitter[i, 0]
        y = reduced_matrix_twitter[i, 1]
        x1 = reduced_matrix_wiki[i, 0]
        y1 = reduced_matrix_wiki[i, 1]      
        ax.annotate(target_word, (x, y))
        ax.annotate(target_word, (x1, y1))

    # Plot.show()
    # Plot.savefig(title +'_word_vectors.png');

def dist(x,y):   
    return np.sqrt(np.sum((x-y)**2))