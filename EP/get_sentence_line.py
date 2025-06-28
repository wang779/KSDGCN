import path as p

# 获取句子的行号（第几个句子）
def get_sen_line(file):
    with open(file, 'r', encoding='utf-8', newline='\n', errors='ignore') as fin:
        lines = fin.readlines()
    dic = {}
    for i in range(0, len(lines), 2):
        dic[lines[i].strip().lower()] = int(i / 2)
    return dic


train_sen_line = get_sen_line(p.dataset_path + "/final_train.raw")
test_sen_line = get_sen_line(p.dataset_path + "/final_test.raw")
