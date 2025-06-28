# -*- coding:utf-8 -*-


from tqdm import tqdm
import random
import argparse
import torch


def load_knowledge_data():
    head_cluster, tail_cluster = {}, {}
    num_del = total = 0
    with open(r"/home/jd/code/yz/gat/work2_wyj/wikidata5m_all_triplet.txt", 'r', encoding='utf-8') as fin:
        lines = fin.readlines()
        for i in tqdm(range(len(lines))):
            line = lines[i]
            v = line.strip().split("\t")
            if len(v) != 3:
                continue
            h, r, t = v
            if h in head_cluster:
                head_cluster[h].append((r, t))
            else:
                head_cluster[h] = [(r, t)]
            if t in tail_cluster:
                tail_cluster[t].append((r, h))
            else:
                tail_cluster[t] = [(r, h)]
            total += 1
    print('wikidata5m_triplet.txt (Wikidata5M) loaded!')
    print('deleted {} triples from Wikidata5M.'.format(num_del))
    return head_cluster, tail_cluster


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract_Triple.")
    parser.add_argument('--dataset_str', type=str, default='Ciao', help='Data set. Default is Ciao.')
    args = parser.parse_args()

    path = r"/home/jd/code/yz/datasets/reddic_tagme阈值0/entity_2_wikiid_train.txt"
    head_cluster, tail_cluster = load_knowledge_data()

    id_2_wikiid = {}
    max_neighbors = 1
    with open(path, 'r', encoding='utf-8') as f1:
        for line in f1:
            if len(line.strip('\n').split("\t")) != 4:
              print(line)
              continue
            product_id, entity_wiki_id, mention, entity = line.strip('\n').split("\t")
            if product_id not in id_2_wikiid:
                id_2_wikiid[product_id] = []
            id_2_wikiid[product_id].append(entity_wiki_id)

    # num_entity, num_rea = 0, 0
    # releation = set()
    #triple = set()
    triple = []
    # total_entity = set()

    for key, values in id_2_wikiid.items():
        print("id_2_wikiid.items():", key, values)
        for value in values:
        
            # if value in head_cluster and random.uniform(0, 1) > 0.5:
            if value in head_cluster:
                # print("value in head_cluster and random.uniform(0, 1) > 0.5:", value)
                print("value:", value)
                triple_lst = head_cluster[value]
                print("head_cluster[value]:", triple_lst)
                head_neighbors = 0
                shuffled_triple = random.sample(triple_lst, max_neighbors)
                print("shuffled_triple:", shuffled_triple)
                for (r, t) in shuffled_triple:
                    # if head_neighbors >= max_neighbors:
                    #     break
                    # head_neighbors += 1

                    # triple.add((key, value, r, t))
                    triple.append((key, value, r, t))
                    # total_entity.add(r)
                    # releation.add(t)

    print("提取出的triple数量：", len(triple))

    path_out = r"/home/jd/code/yz/datasets/reddic_tagme阈值0/triple_train.txt"

    with open(path_out, 'w', encoding='utf-8') as ftrain:
        ftrain.writelines(str(len(triple)) + "\n")
        for key in triple:
            id, entity_1, rel, entity_2 = key
            ftrain.writelines(str(id) + "\t" + str(entity_1) + "\t" + str(rel) + "\t" + str(entity_2) + "\n")
    print("extract triple successfully!")
