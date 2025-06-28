def read_and_extract(file_path, entity_output_file, relation_output_file):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    heads_entity = []
    tails_entity = []
    relations = []

    for line in lines:
        columns = line.strip().split('\t')
        if len(columns) == 4:
            head = columns[1]
            tail = columns[3]
            relation = columns[2]
            heads_entity.append(head)
            tails_entity.append(tail)
            relations.append(relation)

    with open(entity_output_file, 'w', encoding='utf-8') as output_file:
        for entity in heads_entity:
            output_file.write(f"{entity}\n")
        for entity in tails_entity:
            output_file.write(f"{entity}\n")
    with open(relation_output_file, 'w', encoding='utf-8') as output_file:
        for relation in relations:
            output_file.write(f"{relation}\n")

if __name__ == "__main__":
    input_file = r"/home/jd/code/yz/datasets/reddic_tagme阈值0/final_train_triple.txt"  
    entity_output_file = r"/home/jd/code/yz/datasets/reddic_tagme阈值0/final_entity_train.txt" 
    relation_output_file = r"/home/jd/code/yz/datasets/reddic_tagme阈值0/final_relation_train.txt"
    read_and_extract(input_file, entity_output_file, relation_output_file)
