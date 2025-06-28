import tagme
import requests
from difflib import SequenceMatcher

API_ENDPOINT = "https://www.wikidata.org/w/api.php"


# 计算相似度
def similarity(a, b):
    return SequenceMatcher(None, a, b).ratio()


# 读取包含文本句子的文件，每行包含一个句子
def read_sentences(file):
    sentences = []
    with open(file, 'r', encoding='utf-8') as f:
        for line in f:
            sentence = line.strip()
            if sentence:  # 跳过空行
                sentences.append(sentence)
    return sentences


# 使用 TagMe 提取实体
def extract_entities(sentences):
    entities = []
    for sentence in sentences:
        ann_generator = tagme.annotate(sentence)
        sentence_entities = []
        for annotation in ann_generator.get_annotations(0.2):
            mention = annotation.mention
            entity_title = annotation.entity_title
            sentence_entities.append((mention, entity_title))
        entities.append(sentence_entities)
    return entities


# 获取实体的 Wikidata ID
def get_entity_id(entity_title):
    params = {
        'action': 'wbsearchentities',
        'format': 'json',
        'language': 'en',
        'search': entity_title
    }
    r = requests.get(API_ENDPOINT, params=params).json().get('search', [])
    if r:
        return r[0].get("id", "")
    else:
        return None


# 生成关系并写入文件
def generate_relations(entities, output_file):
    with open(output_file, 'w', encoding='utf-8') as fw:
        for i, sentence_entities in enumerate(entities):
            for j in range(len(sentence_entities)):
                for k in range(j + 1, len(sentence_entities)):
                    mention1, entity_title1 = sentence_entities[j]
                    mention2, entity_title2 = sentence_entities[k]

                    # 计算实体描述的相似度
                    mention1_des = mention1.replace(' ', '_').replace('-', '_').replace('.', '').replace('(',
                                                                                                         '').replace(
                        ')', '').lower()
                    mention2_des = mention2.replace(' ', '_').replace('-', '_').replace('.', '').replace('(',
                                                                                                         '').replace(
                        ')', '').lower()

                    if similarity(mention1_des, entity_title1.lower()) > 0.4 and similarity(mention2_des,
                                                                                            entity_title2.lower()) > 0.4:
                        entity_id1 = get_entity_id(entity_title1)
                        entity_id2 = get_entity_id(entity_title2)

                        if entity_id1 and entity_id2:
                            fw.write(f"{i}\t{entity_id1}\tP31\t{entity_id2}\n")
                            fw.flush()


if __name__ == "__main__":
    # 设置 TagMe 的访问令牌
    # tagme.GCUBE_TOKEN = "your_tagme_token_here"
    tagme.GCUBE_TOKEN = ""

    # 读取源文本文件
    source_file = r""  # 替换为实际的源文本文件路径
    sentences = read_sentences(source_file)

    # 提取实体
    entities = extract_entities(sentences)

    # 生成关系并写入文件
    output_file = r""  # 替换为实际的输出文件路径
    generate_relations(entities, output_file)

    print("Extraction and relation generation complete.")
