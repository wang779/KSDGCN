import path as p


def read_files(file, file_type):
    # 存储每个句子的实体数量和每个句子的实体列表。
    l = {}  # 每个句子（有实体的）的实体数量
    line_ent = {}
    with open(file, 'r', encoding='utf-8') as fin:
        lines = fin.readlines()
    for line in lines:
        # print(line)
        # 将句子编号num作为字典的键，实体数量作为对应值。同时，将实体h添加到line_ent字典中对应句子编号的值列表中。
        num, h, t, r = line.strip('\n').split('\t')
        if num in l.keys():
            l[num] = l[num] + 1
            line_ent[num].append(h)
        else:
            l[num] = 1
            line_ent[num] = [h]

    # 没有实体的句子
    # 根据文件类型，确定总的句子数量，并通过循环遍历所有句子编号，
    # 如果某个句子编号不在字典`l`的键中，则将其添加到字典中，并将实体数量初始化为0。
    if file_type == 'train':
        sentences = p.datasets_num[0]
    else:
        sentences = p.datasets_num[1]
    for i in range(sentences):
        if str(i) not in l.keys():
            l[str(i)] = 0

    return l, line_ent


# 每个句子的实体数量和实体列表，key是句子行号（第几个句子）
ent_num_per_test_sentence, ent_per_test_sentence = read_files(p.dataset_path + '/final_test_triple.txt', 'test')
ent_num_per_train_sentence, ent_per_train_sentence = read_files(p.dataset_path + '/final_train_triple.txt', 'train')
