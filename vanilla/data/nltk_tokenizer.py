#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 23 14:53:43 2018

@author: grandee
"""

import sys
sys.path.append("/home/grandee/Projects/language model/utility")
import grandee_toolbox as grandee
import nltk
import re

text = grandee.read('seame.mono')
# tokenize using nltk
sent = text.split('\n')
text = []
for line in sent:
    newline = nltk.word_tokenize(line)
    if not newline == []:
        text.append(newline)
grandee.write2dlist(text, 'mono.nltk_tokenizer.txt')

text = grandee.read('mono.nltk_tokenizer.txt')
text = re.sub(r"< unk >", r"<unk>", text)
text = re.sub(r"< num >", r"<num>", text)
grandee.writestring(text, 'mono.nltk_tokenizer.txt')


text = grandee.read('seame.cs')
# tokenize using nltk
sent = text.split('\n')
text = []
for line in sent:
    newline = nltk.word_tokenize(line)
    if not newline == []:
        text.append(newline)
grandee.write2dlist(text, 'cs.nltk_tokenizer.txt')

text = grandee.read('cs.nltk_tokenizer.txt')
text = re.sub(r"< unk >", r"<unk>", text)
text = re.sub(r"< num >", r"<num>", text)
grandee.writestring(text, 'cs.nltk_tokenizer.txt')


# text = read('valid.txt')
# # tokenize using nltk
# sent = text.split('\n')
# text = []
# for line in sent:
#     newline = nltk.word_tokenize(line)
#     if not newline == []:
#         text.append(newline)
# write2dlist(text, 'valid.nltk_tokenizer.txt')
#
# text = read('valid.nltk_tokenizer.txt')
# text = re.sub(r"< unk >", r"<unk>", text)
# text = re.sub(r"< num >", r"<num>", text)
# writestring(text, 'valid.nltk_tokenizer.txt')