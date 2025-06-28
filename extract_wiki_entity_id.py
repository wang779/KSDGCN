import ast
# from wikimapper import WikiMapper
import requests
from difflib import SequenceMatcher

API_ENDPOINT = "https://www.wikidata.org/w/api.php"
import argparse


def similarity(a, b):
    return SequenceMatcher(None, a, b).ratio()


def extract_entity_id(r_file, w_file):
    with open(w_file, 'w', encoding='utf-8') as fw:
        with open(r_file, encoding='utf-8') as f:
            for line in f:
                if len(line.strip("\n").split('\t')) != 3:
                    continue
                id, mention, entity = line.strip("\n").split('\t')

                entity_des = entity.replace(' ', '_').replace('-', '_').replace('.', '').replace('(', '').replace(')',
                                                                                                                  '').lower()
                mention_des = mention.replace(' ', '_').replace('-', '_').replace('.', '').replace('(', '').replace(')',
                                                                                                                    '').lower()

                if similarity(mention_des, entity_des) > 0.1:
                    description_list = []
                    x = entity
                    params = {
                        'action': 'wbsearchentities',
                        'format': 'json',
                        'language': 'en',
                        'search': x
                    }
                    r = requests.get(API_ENDPOINT, params=params).json().get('search', "")

                    if r:
                        entity_id = r[0].get("id", "")
                        description_list.append(r[0].get("description", ""))

                        fw.write(id + "\t" + entity_id + '\t' + mention + "\t" + x + '\n')
                        fw.flush()
                    else:
                        description_list.append("")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Knowledge_Trust_Assessment.")
    parser.add_argument('--dataset_str', type=str, default='Ciao', help='Data set. Default is Ciao.')
    args = parser.parse_args()

    r1 = r"/home/jd/code/yz/datasets/reddic/product_tagMe_train.txt"
    w2 = r"/home/jd/code/yz/datasets/reddic/entity_2_wikiid_train.txt"

    # r1 = "/data1/Yuzz/Trust_Model/data/{}/new_data/KG/product_tagMe.txt".format(args.dataset_str)
    # w2 = "/data1/Yuzz/Trust_Model/data/{}/new_data/KG/entity_2_wikiid.txt".format(args.dataset_str)

    extract_entity_id(r1, w2)
