import platform

local_dataset_path = r'D:\TJU科研\KSDGCN_code备份\work2\datasets1\reddic'
remote_dataset_path = r'/home/jd/code/yz/datasets1/reddic'

# 不同算法中train和test数量的划分
datasets_n = [[1333, 158],
                [8497, 1063],
                [42717, 10913],
                [20842, 5211]]

# 定义了一个二维列表know_select_algorithm，其中包含三个子列表，每个子列表表示不同的知识选择算法对应的训练集和测试集文件名。
know_select_algorithm = [['train_selectedKnow_minor.txt', 'test_selectedKnow_minor.txt'],
                         ['train_selectedKnow_major.txt', 'test_selectedKnow_major.txt'],
                         ['train_selectedKnow_contrast.txt', 'test_selectedKnow_contrast.txt']]

# local_bert = '/Users/eleanoryu/Downloads/code/bert-base-cased'
local_bert = r'D:\下载\bert-base-uncased'
remote_bert = '/home/jd/code/yz/bert-base-cased'

# 第几个数据集
datasets_num = datasets_n[3]
# 第几个知识选择算法
know_file = know_select_algorithm[1]

if 'Windows' == platform.system():
    dataset_path, bert_path = local_dataset_path, local_bert
elif 'Linux' == platform.system():
    dataset_path, bert_path = remote_dataset_path, remote_bert
