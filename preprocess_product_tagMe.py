import tagme
import argparse


# 读取实体映射文件，将实体名称映射到维基数据ID
def read_entity_map(file):
    # Read entity map: map entityName to QID
    ent_map = {}
    with open(file, encoding='utf-8') as fin:
        for line in fin:
            name, qid = line.strip().split("\t")
            ent_map[name] = qid
    return ent_map


# # 读取包含实体名称的文件,每行包含了实体的名称、数量和类别等信息
# def read_files(file):
#     lines = []
#     with open(file, encoding='utf-8') as f:
#         for line in f:
#             name, num, cat = line.strip("").split("\t")
#             lines.append([name])
#     return lines


def read_files(file):
    lines = []

    with open(file, encoding='utf-8') as f:
        text = f.readlines()
        for i in range(0, len(text), 2):
            sentence = text[i].strip()
            lines.append([sentence])
    return lines


# 用于将实体抽取结果写入文件，使用 TagMe 工具抽取实体。
def write_files_entry(tagme, lines, w_file):
    dict_entity = {}
    with open(w_file, 'w', encoding='utf-8') as fw:
        for i, line in enumerate(lines):
            print(i)
            mention_list, mention_offset_list, entry_list, entry_id_list, score_list = [], [], [], [], []
            if line == ['']:
                row = [str(i), str(mention_list), str(entry_list), str(mention_offset_list), str(line[0]),
                       str(score_list)]
                fw.write('\t'.join(row) + '\n')
                fw.flush()
                continue
            dict_entity_id = {}
            ann_generator = tagme.annotate(line[0])
            # 初始默认0.2
            for x in ann_generator.get_annotations(0):
                mention = x.mention
                charspan = (x.begin, x.end)
                entity_title = x.entity_title

                fw.write(str(i) + "\t" + mention + "\t" + entity_title + "\n")
                fw.flush()

                if entity_title not in dict_entity.keys():
                    dict_entity[entity_title] = 1
                else:
                    dict_entity[entity_title] = dict_entity[entity_title] + 1

    print("Success!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Knowledge_Trust_Assessment.")

    parser.add_argument('--dataset_str', type=str, default='Ciao', help='Data set. Default is Ciao.')
    args = parser.parse_args()

    r1 = r"/home/jd/code/yz/datasets/reddic_tagme阈值0/final_test.raw"
    w1 = r"/home/jd/code/yz/datasets/reddic_tagme阈值0/product_tagMe_test.txt"

    # r1 = "/data1/Yuzz/Trust_Model/data/{}/new_data/product.txt".format(args.dataset_str)
    # w1 = "/data1/Yuzz/Trust_Model/data/{}/new_data/KG/product_tagMe.txt".format(args.dataset_str)

    # # Use TAGME
    # Set the authorization token for subsequent calls.
    # 账号
    tagme.GCUBE_TOKEN = ""
    lines = read_files(r1)
    write_files_entry(tagme, lines, w1)
