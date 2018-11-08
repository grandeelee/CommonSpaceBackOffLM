from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


# the monolingual embedding is from the base line language model.
# load trained embedding [vocab_n, feature_dim]
filename = "data/model_300_unk_embedding.npy"
embedding = np.load(filename)

for n in [100]:
	# perform kmean cluster of 500
	kmeans = KMeans(n_clusters=n, verbose=1).fit(embedding)
	# get label of training embedding [vocab_n, ]
	labels = kmeans.labels_
	np.save("data/monolingual_embedding/{}_{}c_labels".format(filename.split("/")[-1], n), labels)
	# get centers of kmean [n_clusters, feature_dim]
	clusters = kmeans.cluster_centers_
	# get centers according to the index of original embedding [vocab_n, feature_dim]
	# classes = []
	# for idx in labels:
	# 	classes.append(clusters[idx])

	np.save("data/monolingual_embedding/{}_{}c_embed".format(filename.split("/")[-1], n), clusters)

	# testing
	# test_embedding = embedding[:100]
	# test_clusters = classes[:100]
	# print(cosine_similarity(test_clusters, test_embedding))



