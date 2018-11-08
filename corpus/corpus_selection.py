import os


folder_path = [
                # "casict2015",
              # "news commentary",
              # "OpenSub",
              "TED2013",
              # "umcorpus"
               ]

en = ""
zh = ""

for path in folder_path:
    path_en = os.path.join(path, "tokenized_en")
    path_zh = os.path.join(path, "tokenized_zh")
    with open(path_en, encoding='utf8') as infile:
        en = en + infile.read()
    with open(path_zh, encoding='utf8') as infile:
        zh = zh + infile.read()

with open('raw_convo_small.en', 'w' , encoding='utf8') as file:
    file.writelines(en)

with open('raw_convo_small.zh', 'w' , encoding='utf8') as file:
    file.writelines(zh)