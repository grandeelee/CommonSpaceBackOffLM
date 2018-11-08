# This script extract CS sentences from test set

import re

# Read in data
with open("test.nltk_tokenizer.txt", "r") as f:
	text = f.read().split('\n')

cstext = []
monotext = []
for line in text:
	cols = re.findall(r"[\u4e00-\u9fff]+", line)
	if not cols:
		# this is all the english monolingual
		monotext.append(line)
	else:
		cols = re.findall(r"[a-z]+", line)
		if not cols:
			# this is all the chinese monolingual
			monotext.append(line)
		else:
			cstext.append(line)

# write to file
with open("test.nltk_tokenizer.txt.cs", "w") as f:
	f.writelines('\n'.join([line for line in cstext]))

with open("test.nltk_tokenizer.txt.mono", "w") as f:
	f.writelines('\n'.join([line for line in monotext]))

print("total sentence length: %d, cs+mono length: %d" %(len(text), len(cstext)+len(monotext)))