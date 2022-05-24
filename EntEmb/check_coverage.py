import json
import os
from models.common.utils import generate_text_mentions_data, load_json
import models.common.conifgs as configs

def load_fgs2ee_entity():
    enti_file_dir = './data/entities_types_texts'
    enti_file_list = ['single_entities.ndjson', 'picked_entities.ndjson']
    entities = set()
    for fi in enti_file_list:
        fid = os.path.join(enti_file_dir, fi)
        with open(fid, 'r') as fp:
            for line in fp:
                cont = json.loads(line)
                entities.add(cont[0])
    print('FGS2EE entity number:', len(entities))
    return entities

def load_ganea_entity():
    entities = set()
    enti_file_dir = './data/ganea/generated/embeddings/word_ent_embs/dict.entity'
    with open(enti_file_dir, 'r') as fp:
        for line in fp:
            cont = line.split('\t')
            ent = cont[0][22:]
            entities.add(ent)
    print('Ganea entity number:', len(entities))
    return entities

def load_cqa_entity(get_id=False):
    cqa_data = load_json(configs.CQAEL_DATASET_PATH)
    entities = set()
    start = 1 if get_id else 0
    for question in cqa_data['questions']:
        for mention in question['mentions']:
            if mention['Gold_index'] != -1:
                entities.update(mention['Candidates'].split('\t')[start::3])

        for answer in question['answers']:
            for mention in answer['mentions']:
                if mention['Gold_index'] != -1:
                    entities.update(mention['Candidates'].split('\t')[start::3])
    print('CQA entity number:', len(entities))
    return entities


def load_deeped_entities():
    deeped_entities = set()
    with open(configs.CQAEL_EMBEDDING_PATH, 'r') as fp:
        for line in fp:
            line = line.strip()
            deeped_entities.add(line.split('\t')[0])
    print('Deeped entity number:', len(deeped_entities))
    return deeped_entities


if __name__ == '__main__':
    cqa_entities = load_cqa_entity()
    cqa_entities_ids = load_cqa_entity(get_id=True)
    deeped_entities_ids = load_deeped_entities()

    with open(os.path.join(configs.DATASET_ROOT_PATH, 'cqa_cand_entities.json'), 'w') as f:
        json.dump(list(cqa_entities_ids), f)
        exit()

    fgs2ee_entities = load_fgs2ee_entity()
    ganea_entities = load_ganea_entity()

    print('CQA and FGS2EE intersection:', len(cqa_entities & fgs2ee_entities))
    print('CQA and Ganea intersection:', len(cqa_entities & ganea_entities))
    print('CQA and Deep-ED intersection:', len(cqa_entities_ids & deeped_entities_ids))


