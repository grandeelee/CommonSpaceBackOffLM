import numpy as np
import matplotlib.pyplot as plt
import scipy.io as matlabsave
import matplotlib.font_manager as mfm

font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Light.ttc"
prop = mfm.FontProperties(fname=font_path)


entropy = np.load("embed_class_discussion_expt1/testtest_sentence_entropy.npy")
sentence = np.load("embed_class_discussion_expt1/testtest_sentence.npy")

# original_entroy = np.load("tmp/model_300_unktest_sentence_entroy.npy")

for i, name in enumerate(["a", "b","c", "d"]):
	n_words = 500
	entropy_tmp = np.array(entropy[i*n_words:i*n_words+n_words], dtype=float)
	# comment two line below if not for difference plotting
	# diff_entropy_tmp = np.array(original_entroy[i*n_words:i*n_words+n_words])-entropy_tmp
	# entropy_tmp = np.log(np.maximum(diff_entropy_tmp, [1e-5 for _ in range(n_words)]))
	sentence_tmp = sentence[i*n_words:i*n_words+n_words]

	entropy_tmp = np.reshape(entropy_tmp, (-1, 50))
	sentence_tmp = np.reshape(sentence_tmp, (-1, 50))

	col = len(entropy_tmp)
	print(col)

	fig, axs = plt.subplots(figsize=(20, col), nrows=col)

	# # plot just the positive data and save the
	# # color "mappable" object returned by ax1.imshow
	for idx, axis in enumerate(axs):
		axis.set_yticks([])
		axis.set_xticks(np.arange(len(entropy_tmp[idx])))
		pos = axis.imshow([entropy_tmp[idx]], cmap='Blues', interpolation='none')
		axis.set_xticklabels(sentence_tmp[idx], rotation=45, fontproperties=prop)
		# for xidx, word in enumerate(sentence[idx]):
		# 	plt.text(xidx, idx-5, s=word, ha="center", va="center", fontproperties=prop)

	# # add the colorbar using the figure's method,
	# # telling which mappable we're talking about and
	# # which axes object it should be near
	fig.colorbar(pos, ax=axs, orientation='vertical', fraction=.1)

	# # repeat everything above for the negative data
	# neg = ax2.imshow(Zneg, cmap='Reds_r', interpolation='none')
	# fig.colorbar(neg, ax=ax2)
	fig.savefig("with_class_{}.png".format(name))
	plt.close()

