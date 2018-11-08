import nltk
import jieba
import re
import os

# format all the corpus into the proper format
def en_tokenizer(source, output):
    print("starting tokenizing " + source)
    # for english
    with open(source, encoding='utf8') as infile:
        sent = infile.read().lower().split("\n")
    # tokenize using nltk
    text = []
    for line in sent:
        # remove punctunoation
        line = ' '.join(re.findall(r"[a-z'\u4e00-\u9fff]+", line))
        newline = nltk.word_tokenize(line)
        text.append(newline)
    with open(output, 'w', encoding='utf8') as file:
        file.writelines(' '.join(str(j) for j in i) + '\n' for i in text)
    print("finished " + source)

def zh_tokenizer(source, output):
    print("starting tokenizing " + source)
    # for chinese
    with open(source, encoding='utf8') as infile:
        sent = infile.read().lower().split("\n")
    # the same dict for SEAME
    jieba.set_dictionary('dict.txt')
    seglist = []
    for line in sent:
        # remove punctuation
        line = ' '.join(re.findall(r"[a-z'\u4e00-\u9fff]+", line))
        seg = ' '.join(jieba.cut(line, cut_all=False, HMM=False))
        seg = ' '.join(re.findall(r"[a-z'\u4e00-\u9fff]+", seg))
        seglist.append(seg)
    with open(output, 'w' , encoding='utf8') as file:
        file.writelines(i + '\n' for i in seglist)
    print("finished " + source)

folder_path = ["casict2015",
              "news commentary",
              "OpenSub",
              "TED2013",
              "umcorpus"]

for path in folder_path:
    source_path_en = os.path.join(path, "en")
    source_path_zh = os.path.join(path, "zh")
    target_path_en = os.path.join(path, "tokenized_en")
    target_path_zh = os.path.join(path, "tokenized_zh")
    en_tokenizer(source_path_en, target_path_en)
    zh_tokenizer(source_path_zh, target_path_zh)