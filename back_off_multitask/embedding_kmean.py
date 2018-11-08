from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


# load trained embedding [vocab_n, feature_dim]
embedding = np.load("data/embedding_300dim.npy")

for n in [500, 700, 900, 1100, 1300, 1500]:
	# perform kmean cluster of 500
	kmeans = KMeans(n_clusters=n, verbose=1).fit(embedding)
	# get label of training embedding [vocab_n, ]
	labels = kmeans.labels_
	# get centers of kmean [n_clusters, feature_dim]
	clusters = kmeans.cluster_centers_
	# get centers according to the index of original embedding [vocab_n, feature_dim]
	classes = []
	for idx in labels:
		classes.append(clusters[idx])

	np.save("data/embedding_kmean_300dim_{}".format(n), classes)

	# testing
	# test_embedding = embedding[:100]
	# test_clusters = classes[:100]
	# print(cosine_similarity(test_clusters, test_embedding))



