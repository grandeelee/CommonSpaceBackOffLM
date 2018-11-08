import glob
import re


def readmanytext(filename):
	'This function takes in filepath such as "./phaseII/*.txt" and return all text in it'
	text = ''
	for filename in glob.glob("./phaseII/*.txt"):
		with open(filename, encoding='utf8') as infile:
			text = text + infile.read().lower();
	return text


def read(filename):
	with open(filename, 'r', encoding='utf8') as infile:
		text = infile.read()
	return text


def writestring(variable, filename):
	with open(filename, 'w', encoding='utf8') as file:
		file.writelines(variable)


def writelist(variable, filename):
	with open(filename, 'w', encoding='utf8') as file:
		file.writelines('\n'.join(i for i in variable))


def write2dlist(variable, filename):
	with open(filename, 'w', encoding='utf8') as file:
		file.writelines(' '.join(str(j) for j in i) + '\n' for i in variable)


def writenospace(variable, filename):
	with open(filename, 'w', encoding='utf8') as file:
		file.writelines(''.join(str(j) for j in i) + '\n' for i in variable)


def readdict(filename):
	with open(filename, encoding='utf8') as infile:
		text = infile.read()
	sent = text.split('\n')
	out = [line.split(' ')[0] for line in sent]
	return out

def get_OOV(source, target, oov_path):
	'get list of word in source not found in target stored in oov_path'
	with open(source, encoding='utf8') as infile:
		source_vocab = infile.read().split()
	with open(target, encoding='utf8') as infile:
		target_vocab = infile.read().split()
	oov = []
	for word in source_vocab:
		if word not in target_vocab:
			oov.append(word)
	writelist(oov, oov_path)

