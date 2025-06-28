# 去停用词，去标点

import tqdm
import path as p
import spacy
from nltk.corpus import stopwords
import string

punc = string.punctuation
nlp = spacy.load('en_core_web_sm')
stop = stopwords.words('english')


def is_special_word(str):
    if str in punc:
        return True
    elif str == ' ':
        return True
    elif str == '\n':
        return True
    elif str == '  ':
        return True
    elif str == '  ':
        return True
    elif str == '    ':
        return True
    elif str == '     ':
        return True
    elif str == '\t':
        return True
    elif str == '#':
        return True
    elif str == '@':
        return True
    elif str == '\'':
        return True
    else:
        return False


def remove_stopwords(fi, fo):
    fin = open(fi, 'r', encoding='utf-8')
    lines = fin.readlines()
    print(len(lines))
    fin.close()
    fout = open(fo, 'w', encoding='utf-8')

    for i in range(0, len(lines), 2):
        text = lines[i].lower().strip()
        words = nlp(text)
        word_list = [str(x.lemma_) for x in words]
        content = " ".join([word for word in word_list if word not in stop and not is_special_word(word)])
        fout.write(content)
        fout.write('\n')
        j = i + 1
        # print("j:",j)
        fout.write(lines[j])


# 将.raw文件生成final_train.raw文件
if __name__ == '__main__':
    remove_stopwords(p.dataset_path + '/train.raw', p.dataset_path + '/final_train.raw')
    
    # print(p.dataset_path + '/test.raw')
    remove_stopwords(p.dataset_path + '/test.raw', p.dataset_path + '/final_test.raw')
