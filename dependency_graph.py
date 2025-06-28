# -*- coding: utf-8 -*-
import path as p
import numpy as np
import spacy
import pickle
import json
from tqdm import tqdm

nlp = spacy.load('en_core_web_sm')

bbb = {}


# 生成给定文本的依赖关系邻接矩阵。该函数接受一个字符串参数text，表示待处理的文本。
def dependency_adj_matrix(text):
    # https://spacy.io/docs/usage/processing-text
    # 对输入文本进行处理，得到Spacy的文档对象。
    document = nlp(text)
    seq_len = len(document)
    bbb[text] = []
    matrix = np.zeros((seq_len, seq_len)).astype('float32')
    for token in document:
        bbb[text].append(token.text)
        matrix[token.i][token.i] = 1
        for child in token.children:
            matrix[token.i][child.i] = 1
            matrix[child.i][token.i] = 1
    return matrix


# 用于处理文件并生成依赖关系邻接矩阵数据
def process(filename):
    with open(filename, 'r', encoding='utf-8', newline='\n', errors='ignore') as fin:
        lines = fin.readlines()
    # # 初始化一个空字典idx2graph，用于存储每个文本对应的依赖关系邻接矩阵。
    idx2graph = {}

    # .graph.new存储依赖图的邻接矩阵
    with open(filename + '.graph.new', 'wb') as fout:
        for i in tqdm(range(0, len(lines), 2)):
            text = lines[i].lower().strip()
            adj_matrix = dependency_adj_matrix(text)
            idx2graph[i] = adj_matrix
        # 使用pickle模块将idx2graph字典序列化，并写入到打开的文件中，以便后续读取。
        pickle.dump(idx2graph, fout)


# 此脚本读取.raw文件生成依赖图的邻接矩阵，写到文档.graph.new中
if __name__ == '__main__':
    process(p.dataset_path + '/final_train.raw')
    process(p.dataset_path + '/final_test.raw')

    json_string = json.dumps(bbb)
    f = open(p.dataset_path + '/nlp1.json', "w", encoding="utf-8")
    f.write(json_string)
    f.close()
