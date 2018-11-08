from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re

# load trained embedding [vocab_n, feature_dim]
embedding = np.load("data/embedding.npy")
with open("data/pseudo_cs_data_small_with_mono_vocab", "r") as f:
	vocab = f.read().split()
for n in [1000]:
	# perform kmean cluster of 500
	kmeans = KMeans(n_clusters=n, verbose=1).fit(embedding)
	# get label of training embedding [vocab_n, ]
	labels = kmeans.labels_
	# for printing and storing as word file
	cluster_string = ["" for _ in range(1000)]
	for idx, word in enumerate(vocab):
		# if the word is chinese, its cluster add 500
	# 	if re.findall(r"[a-z']+", word) == []:
	# 		labels[idx] += 500
		cluster_string[labels[idx]] += (word + " ")
	# labels map word id to cluster id of range [0, 1000)
	np.save("data/pseudo_CS_data_small_with_mono_wordid2clusterid_{}".format(n), labels)
	# get centers of kmean [n_clusters, feature_dim]
	clusters = kmeans.cluster_centers_
	# get centers according to the index of original embedding [vocab_n, feature_dim]
	# classes = []
	# for idx in labels:
	# 	classes.append(clusters[idx])

	np.save("data/pseudo_CS_data_small_with_mono_kmean_300dim_{}c".format(n), clusters)
	with open("data/pseudo_cs_data_small_with_mono_cluster", "w") as f:
		f.writelines("\n".join(i for i in cluster_string))
	# # testing
	# # test_embedding = embedding[:100]
	# # test_clusters = classes[:100]
	# # print(cosine_similarity(test_clusters, test_embedding))

# also prepare the training and testing data


