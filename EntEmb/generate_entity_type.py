import argparse
import json
import os

import numpy as np

"""
this file is for geneate fine-grained semantic entity embeddings, which
is the average of Word2Vec embeddings of type words.
"""


def load_type_vec(dic_dir, vec_dir):
    """
    dict is for indexing type words,
    type_vec is from google Word2Vec
    """
    type_vec = np.load(vec_dir)
    type2id = {}
    id2type = []
    with open(dic_dir, 'r') as fp:
        for line in fp:
            cont = line.split('\t')
            typ = cont[0]
            type2id[typ] = len(id2type)
            id2type.append(typ)
    return type2id, type_vec


def load_type_words(type_words_file, oov_type_words_remap_file):
    # type_words, oov_type_words_remap
    type_words = {}
    oov_type_words_remap = {}
    with open(type_words_file, 'r') as fp:
        for line in fp:
            cont = eval(line)
            type_words[cont[0]] = cont[0]
    with open(oov_type_words_remap_file, 'r') as fp:
        for line in fp:
            cont = eval(line)
            oov_type_words_remap[cont[0]] = cont[1]
    for k, v in oov_type_words_remap.items():
        type_words[v] = k

    return type_words

def get_entity_emb(enti_file_dir, enti_file_list, type2id, type_vec, tee, saving_path):
    """
    generate semantic entity embeddings
    """
    tn, ed = type_vec.shape
    ent_indx = []
    ent_vecc = []
    num = 0
    for fi in enti_file_list:
        fid = os.path.join(enti_file_dir, fi)
        with open(fid, 'r') as fp:
            for line in fp:
                num += 1
                if num % 1000 == 0:
                    print(num)
                cont = json.loads(line)
                e_name = cont[0]
                ot_list = cont[1]
                nt_list = []
                for ot in ot_list:
                    if ot not in type2id.keys():
                        bt = ot.split(' ')
                        for nt in bt:
                            if nt not in nt_list:
                                nt_list.append(nt)
                    else:
                        nt_list.append(ot)
                # use the nt_list
                # print(type_vec[0].shape)
                e_vec = np.zeros(ed)
                tn = 0
                for nnt in nt_list[:tee]:
                    if nnt in type2id.keys():
                        e_vec = e_vec + np.array(type_vec[type2id[nnt]])
                        tn += 1
                    else:
                        print('not in type Voc===' + nnt)
                type_num = np.ones(ed) * tn
                e_vec = e_vec / type_num
                # print(e_vec.shape)
                ent_indx.append(e_name)
                ent_vecc.append(e_vec)
    ent_vecc = np.array(ent_vecc)
    # print(ent_vecc.shape)
    print("Extracted {} entities".format(len(ent_indx)))
    with open(saving_path + '/dict_tee{}.entity'.format(tee), 'w') as ft:
        for ent in ent_indx:
            ft.write('en.wikipedia.org/wiki/' + ent + '\t11\n')
    np.save(saving_path + '/entity_vec_tee{}.npy'.format(tee), ent_vecc)


def get_entity_types(enti_wikipedia_text, type2id, type_words, saving_path):
    # load dict from file
    enti_text_list = []
    with open(enti_wikipedia_text, 'r') as fp:
        for line in fp:
            enti_text_list.append(eval(line))
    formatted_enti_list = []
    for enti in enti_text_list:
        title = enti['title']
        text = enti['text']
        types = set()
        for word in text.split(' '):
            if word in type_words.keys():
                types.add(type_words[word])
        types = list(types)
        formatted_enti_list.append([title, types, text])
    with open(saving_path + '/cqa_entities.ndjson', 'w') as fp:
        for enti in formatted_enti_list:
            fp.write(json.dumps(enti) + '\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--type_dict', type=str, default='./data/type_dict.type')
    parser.add_argument('--type_words', type=str, default='./data/type_list.ndjson')
    parser.add_argument('--oov_type_words_remap', type=str, default='./data/type_list_OOVs_remap.ndjson')
    parser.add_argument('--type_vec', type=str, default='./data/type_vec.npy')
    parser.add_argument('--enti_wikipedia_text', type=str, default='../../dataset/cqa-el/cqa_cand_wikipedia_78559.txt')
    parser.add_argument('--saving_path', type=str, default='./output')
    args = parser.parse_args()

    type_dict = args.type_dict
    type_vec = args.type_vec
    enti_wikipedia_text = args.enti_wikipedia_text
    type_words_file, oov_type_words_remap_file = args.type_words, args.oov_type_words_remap
    saving_path = args.saving_path
    type2id, type_vec = load_type_vec(type_dict, type_vec)
    type_words = load_type_words(type_words_file, oov_type_words_remap_file)
    get_entity_types(enti_wikipedia_text, type2id, type_words, saving_path)
