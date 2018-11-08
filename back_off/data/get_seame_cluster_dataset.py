import numpy as np

output_filename = "model_300_unk"
classes = 100
clusters = np.load("monolingual_embedding/model_300_unk_embedding.npy_100c_labels.npy")


print(len(clusters))
with open("model_300_unk_embedding_vocab", "r") as f:
	vocab =  f.read().split()
print(len(vocab))
word_to_id = dict(zip(vocab, range(len(vocab))))
id_to_cluster = dict(zip(range(len(clusters)), clusters))

with open("train.nltk_tokenizer.txt", "r") as f:
	train_text =  f.read().replace("\n", " </s> ").split()
with open("valid.nltk_tokenizer.txt", "r") as f:
	valid_text =  f.read().replace("\n", " </s> ").split()
with open("test.nltk_tokenizer.txt", "r") as f:
	test_text =  f.read().replace("\n", " </s> ").split()

train_id = [word_to_id[word] if word in word_to_id else word_to_id['<unk>'] for word in train_text]
valid_id = [word_to_id[word] if word in word_to_id else word_to_id['<unk>'] for word in valid_text]
test_id = [word_to_id[word] if word in word_to_id else word_to_id['<unk>'] for word in test_text]

train_class = [id_to_cluster[word] for word in train_id]
np.save(output_filename + ".train.nltk_tokenizer.txt.{}class".format(classes), train_class)
valid_class = [id_to_cluster[word] for word in valid_id]
np.save(output_filename + ".valid.nltk_tokenizer.txt.{}class".format(classes), valid_class)
test_class = [id_to_cluster[word] for word in test_id]
np.save(output_filename + ".test.nltk_tokenizer.txt.{}class".format(classes), test_class)

assert len(train_class) == len(train_id)
print(train_id[:50])
print(train_class[:50])

