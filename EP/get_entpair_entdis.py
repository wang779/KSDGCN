import EP.final_transE as f
import EP.statistic as s
import numpy as np
import torch
'''
定义了一个类 EP，表示实体对。该类具有两个属性：distance 表示实体对之间的距离，ent_pair_emb 表示实体对的嵌入向量。
定义了函数 concat(emb1, emb2)，用于将两个嵌入向量拼接在一起。
定义了函数 distance(emb1, emb2)，用于计算两个嵌入向量之间的距离。
定义了函数 get_entpair_entdistance(dataset_type)，用于获取实体对的嵌入向量和距离。根据参数 dataset_type 的值，分别处理训练集和测试集。
对于每一行文本，如果文本中的实体数量为0或1，则跳过。如果实体数量为2，则计算这两个实体的嵌入向量和距离，并将其存储到 line_epemb_dic 和 line_epdistance_dic 中。
如果实体数量大于2，则计算所有实体对的嵌入向量和距离，并将其存储到 line_epemb_dic 和 line_epdistance_dic 中。
对于每一行文本，选取距离最大的三对实体对，将其嵌入向量和距离存储到 line_epemb_dic 和 line_epdistance_dic 中。
最后返回 line_epemb_dic 和 line_epdistance_dic。
通过调用 get_entpair_entdistance 函数，得到了训练集和测试集中每一行文本中的实体对的嵌入向量和距离。
'''
line_epemb_dic = {}
line_epdistance_dic = {}


class EP:
    def __init__(self):
        self.distance = 0
        self.ent_pair_emb = 0


def concat(emb1, emb2):
    emb = torch.cat([emb1, emb2], dim=0)
    return emb


def distance(emb1, emb2):
    emb = np.sum(np.abs(emb1 - emb2))
    return emb


def get_entpair_entdistance(dataset_type):
    # 这样就把实体对emb和实体对距离搞出来了
    if dataset_type == 'train':
        for line, entity in s.ent_per_train_sentence.items():
            if len(entity) == 0 or len(entity) == 1:
                continue
            if len(entity) == 2:
                ep_emb = concat(torch.tensor(f.dic_train_ent_emb[entity[0]]),
                                torch.tensor(f.dic_train_ent_emb[entity[1]]))
                ep_dis = distance(f.dic_train_ent_emb[entity[0]], f.dic_train_ent_emb[entity[1]])
                line_epemb_dic[line] = [ep_emb.unsqueeze(0)]
                line_epdistance_dic[line] = [ep_dis]
            else:
                l = len(entity)
                ent_pair = []
                # 计算所有实体对的距离，拼接实体对
                for i in range(l):
                    for j in range(i + 1, l):
                        ep = EP()
                        ep.ent_pair_emb = concat(torch.tensor(f.dic_train_ent_emb[entity[i]]),
                                                 torch.tensor(f.dic_train_ent_emb[entity[j]]))
                        ep.distance = distance(f.dic_train_ent_emb[entity[i]], f.dic_train_ent_emb[entity[j]])
                        ent_pair.append(ep)
                # 排序， 选出前三个距离最大的实体对
                ent_pair.sort(key=lambda e: e.distance, reverse=True)
                line_epemb_dic[line] = [ent_pair[0].ent_pair_emb.unsqueeze(0), ent_pair[1].ent_pair_emb.unsqueeze(0),
                                        ent_pair[2].ent_pair_emb.unsqueeze(0)]
                line_epdistance_dic[line] = [ent_pair[0].distance, ent_pair[1].distance, ent_pair[2].distance]
    else:
        for line, entity in s.ent_per_test_sentence.items():
            if len(entity) == 0 or len(entity) == 1:
                continue
            if len(entity) == 2:
                ep_emb = concat(torch.tensor(f.dic_test_ent_emb[entity[0]]),
                                torch.tensor(f.dic_test_ent_emb[entity[1]]))
                ep_dis = distance(f.dic_test_ent_emb[entity[0]], f.dic_test_ent_emb[entity[1]])
                line_epemb_dic[line] = [ep_emb.unsqueeze(0)]
                line_epdistance_dic[line] = [ep_dis]
            else:
                l = len(entity)
                ent_pair = []
                # 计算所有实体对的距离，拼接实体对
                for i in range(l):
                    for j in range(i + 1, l):
                        ep = EP()
                        ep.ent_pair_emb = concat(torch.tensor(f.dic_test_ent_emb[entity[i]]),
                                                 torch.tensor(f.dic_test_ent_emb[entity[j]]))
                        ep.distance = distance(f.dic_test_ent_emb[entity[i]], f.dic_test_ent_emb[entity[j]])
                        ent_pair.append(ep)
                # 排序， 选出前三个距离最大的实体对
                ent_pair.sort(key=lambda e: e.distance, reverse=True)
                line_epemb_dic[line] = [ent_pair[0].ent_pair_emb.unsqueeze(0), ent_pair[1].ent_pair_emb.unsqueeze(0),
                                        ent_pair[2].ent_pair_emb.unsqueeze(0)]
                line_epdistance_dic[line] = [ent_pair[0].distance, ent_pair[1].distance, ent_pair[2].distance]
    return line_epemb_dic, line_epdistance_dic


train_line_epemb_dic, train_line_epdistance_dic = get_entpair_entdistance('train')
test_line_epemb_dic, test_line_epdistance_dic = get_entpair_entdistance('test')
