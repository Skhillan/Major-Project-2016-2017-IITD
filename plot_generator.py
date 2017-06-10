import pickle
import dim_reduce_plot
import pylab as plt
import cosine_sim
import pylab as Plot

def plot_generator(num_words, start_no, end_no, predicted_vecs, wiki_vecs, twitter_vecs, common_vocab):

	start = start_no
	end = end_no

	# predicted_vectors = pickle.load(open('predicted_vecs_' + str(n_words)+ '_common.p', 'rb'))
	# wiki_vectors = pickle.load(open('wiki_vecs_' + str(n_words)+ '_common.p', 'rb'))
	# twitter_vectors = pickle.load(open('twitter_vecs_' + str(n_words)+ '_common.p','rb'))
	# v_common = pickle.load(open('common_vocab.p','rb'))
	# v_common = pickle.load(open('common_vocab_' + str(n_words)+ '_common.p', 'rb'))


	fig = Plot.figure() # common figure

	dim_reduce_plot.dim_reduce_plot(twitter_vecs, wiki_vecs, common_vocab, 'Untransformed', 1, start, end)
	dim_reduce_plot.dim_reduce_plot(predicted_vecs, wiki_vecs, common_vocab, 'transformed', 2, start, end)

	## uncomment for best of both twitter and predicted
	
	# dim_reduce_plot.corrected_plot(twitter_vectors, wiki_vectors, predicted_vectors, v_common, 'Transformed', 2, start, end)

	Plot.show()
	# uncomment for cosine similarities

	# index = []
	# y = []
	# for i in range(start, end+1):
	# 	index.append(i)
	# 	y.append(cosine_sim.cosine_sim(predicted_vectors[i], wiki_vectors[i]))

	# print y

	# plt.figure()
	# max_y = np.amax(y)
	# min_y = np.amin(y)
	# plt.ylim(min_y,1)
	# plt.scatter(index, y, marker ='s', label = 'cosine_similarities_untransformed')
	# plt.legend(loc = 4)

	# plt.savefig('cosine_similarities_untransformed.png')

def plot_generator_normalized(predicted_vecs, wiki_vecs, twitter_vecs, words_unnorm, words_norm):


	# predicted_vectors = pickle.load(open('predicted_vecs_' + str(n_words)+ '_common.p', 'rb'))
	# wiki_vectors = pickle.load(open('wiki_vecs_' + str(n_words)+ '_common.p', 'rb'))
	# twitter_vectors = pickle.load(open('twitter_vecs_' + str(n_words)+ '_common.p','rb'))
	# v_common = pickle.load(open('common_vocab.p','rb'))
	# v_common = pickle.load(open('common_vocab_' + str(n_words)+ '_common.p', 'rb'))


	fig = Plot.figure() # common figure

	dim_reduce_plot.dim_reduce_plot(twitter_vecs, wiki_vecs, common_vocab, 'Untransformed', 1)
	dim_reduce_plot.dim_reduce_plot(predicted_vecs, wiki_vecs, common_vocab, 'transformed', 2)

	## uncomment for best of both twitter and predicted
	
	# dim_reduce_plot.corrected_plot(twitter_vectors, wiki_vectors, predicted_vectors, v_common, 'Transformed', 2, start, end)

	Plot.show()