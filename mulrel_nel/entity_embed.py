import numpy as np

from models.common.utils import load_json
import models.common.conifgs as configs

wiki_prefix = 'en.wikipedia.org/wiki/'

def load_cqa_embed(File):
    print("Loading CQA Embeddings")
    cqa_embeddings = {}
    with open(File,'r') as f:
        for line in f:
            split_line = line.split()
            entity_id = split_line[0]
            embedding = np.array(split_line[1].split(','), dtype=np.float64)
            cqa_embeddings[entity_id] = embedding
    print(f"{len(cqa_embeddings)} entities loaded!")
    return cqa_embeddings


def load_cqa_entity(File):
    cqa_data = load_json(File)
    entities = []
    ids = []
    for question in cqa_data['questions']:
        for mention in question['mentions']:
            if mention['Gold_index'] != -1:
                entities.extend(mention['Candidates'].split('\t')[::3])
                ids.extend(mention['Candidates'].split('\t')[1::3])

        for answer in question['answers']:
            for mention in answer['mentions']:
                if mention['Gold_index'] != -1:
                    entities.extend(mention['Candidates'].split('\t')[::3])
                    ids.extend(mention['Candidates'].split('\t')[1::3])
    cqa_id_entities = dict(zip(ids, entities))
    print(f"{len(cqa_id_entities)} entities loaded!")
    return cqa_id_entities

if __name__=='__main__':
    cqa_id_entities = load_cqa_entity(configs.CQAEL_DATASET_PATH)
    cqa_embeddings = load_cqa_embed(configs.CQAEL_EMBEDDING_PATH)
    cqa_dict_entity = []
    cqa_entity_embeddings = []
    for id, title in cqa_id_entities.items():
        cqa_dict_entity.append(wiki_prefix+title)
        cqa_entity_embeddings.append(cqa_embeddings[id])
    cqa_entity_embeddings = np.array(cqa_entity_embeddings)
    print(cqa_entity_embeddings.shape)

    np.save(configs.CQAEL_DATASET_ROOT_PATH+'cqa_entity_embeddings.npy', cqa_entity_embeddings)

    with open(configs.CQAEL_DATASET_ROOT_PATH+'cqa_dict.entity', 'w') as save:
        for idx, entity in enumerate(cqa_dict_entity):
            save.write(entity + '\t' + str(idx) + '\n')