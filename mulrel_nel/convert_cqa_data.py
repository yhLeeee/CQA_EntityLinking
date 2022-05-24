import itertools
import json
import os.path

from sklearn.model_selection import KFold
from tqdm import tqdm

import models.common.conifgs as configs
from models.common.utils import get_mention_pos, load_json, add_metadata

def load_cqa_data(cqa_file, metadata=False):
    cqa_data = load_json(cqa_file)

    if metadata:
        path = os.path.join(configs.CQAEL_TRAIN_TEST_METADATA_DATA, "metadata_dict.json")
        if os.path.exists(path):
            with open(path, 'r') as fp:
                metadata_dict = json.load(fp)
        else:
            metadata_dict = add_metadata(cqa_data, K=3)
            with open(path, 'w') as fp:
                json.dump(metadata_dict, fp)

    questions = cqa_data['questions']
    ret_data = []
    ret_data_metadata = []
    doc_num = 0
    for question in questions:
        doc_num += 1
        doc_name = f'cqa{doc_num}\tcqa{doc_num}'
        ###########################
        #     process question    #
        ###########################

        q_text = question['question_title']
        q_ments = question['mentions']

        for q_ment in q_ments:
            mention = q_ment['mention']
            q_ment_start, q_ment_len = get_mention_pos(q_text, mention)
            lctx = q_text[:q_ment_start]
            rctx = q_text[q_ment_start + q_ment_len:]

            candidates = q_ment['Candidates'].split('\t')

            gold_index = q_ment['Gold_index']
            if gold_index == -1:
                continue

            gold_entity_title = candidates[(gold_index - 1) * 3]

            final_cand = []

            if len(candidates) != 0:
                cand_zipped = list(zip(*[iter(candidates)] * 3))
                cand_zipped = sorted(cand_zipped, key=lambda x: eval(x[-1]), reverse=True)
                final_cand = []
                for i, c in enumerate(cand_zipped):
                    final_cand.append(f'{c[1]},{c[2]},{c[0]}')
                    if c[0] == gold_entity_title:
                        gold_index = i
            gold_entity = cand_zipped[gold_index]
            gold_entity = f"GT:\t{gold_index},{gold_entity[1]},{gold_entity[2]},{gold_entity[0]}"

            cand_text = 'CANDIDATES' if len(final_cand) != 0 else 'EMPTYCAND'
            cand_list_text = '\t'.join(final_cand)
            one_sample_text = f"{doc_name}\t{mention}\t{lctx}\t{rctx}\t{cand_text}\t{cand_list_text}\t{gold_entity}"

            ret_data.append(one_sample_text)
            if metadata:
                rctx_metadata = rctx + ' ' + ' '.join(metadata_dict[q_text])
                one_sample_text_metadata = f"{doc_name}\t{mention}\t{lctx}\t{rctx_metadata}\t{cand_text}\t{cand_list_text}\t{gold_entity}"
                ret_data_metadata.append(one_sample_text_metadata)

        ###########################
        #     process answers     #
        ###########################
        for answer in question['answers']:
            a_text = answer['answer_content']
            a_ments = answer['mentions']

            for a_ment in a_ments:
                mention = a_ment['mention']
                a_ment_start, a_ment_len = get_mention_pos(a_text, mention)
                lctx = a_text[:a_ment_start]
                rctx = a_text[a_ment_start + a_ment_len:]

                candidates = a_ment['Candidates'].split('\t')

                gold_index = a_ment['Gold_index']
                if gold_index == -1:
                    continue
                gold_entity_title = candidates[(gold_index-1)*3]

                final_cand = []

                if len(candidates) != 0:
                    cand_zipped = list(zip(*[iter(candidates)] * 3))
                    cand_zipped = sorted(cand_zipped, key=lambda x: eval(x[-1]), reverse=True)
                    final_cand = []
                    for i, c in enumerate(cand_zipped):
                        final_cand.append(f'{c[1]},{c[2]},{c[0]}')
                        if c[0] == gold_entity_title:
                            gold_index = i
                gold_entity = cand_zipped[gold_index]
                gold_entity = f"GT:\t{gold_index},{gold_entity[1]},{gold_entity[2]},{gold_entity[0]}"

                cand_text = 'CANDIDATES' if len(final_cand) != 0 else 'EMPTYCAND'
                cand_list_text = '\t'.join(final_cand)
                one_sample_text = f"{doc_name}\t{mention}\t{lctx}\t{rctx}\t{cand_text}\t{cand_list_text}\t{gold_entity}"

                ret_data.append(one_sample_text)

                if metadata:
                    rctx_metadata = rctx+' '+ ' '.join(metadata_dict[a_text])
                    one_sample_text_metadata = f"{doc_name}\t{mention}\t{lctx}\t{rctx_metadata}\t{cand_text}\t{cand_list_text}\t{gold_entity}"
                    ret_data_metadata.append(one_sample_text_metadata)

    print(f"{len(ret_data)} samples in total")
    return ret_data, ret_data_metadata


def split_train_test_data(data, data_metadata,K):
    data_questions = {}
    data_questions_metadata = {}

    for ind, line in enumerate(data):
        doc_name, _ = line.split('\t', 1)
        if doc_name not in data_questions:
            data_questions[doc_name] = []
        data_questions[doc_name].append(line)

        line = data_metadata[ind]
        doc_name_metadata, _ = line.split('\t', 1)
        assert doc_name_metadata == doc_name
        if doc_name not in data_questions_metadata:
            data_questions_metadata[doc_name] = []
        data_questions_metadata[doc_name].append(line)

    kf = KFold(n_splits=K)
    questions = list(data_questions.keys())
    index = 0
    for train_index, test_index in kf.split(questions):
        train_data =  list(itertools.chain.from_iterable([data_questions[questions[t]] for t in train_index]))
        test_data = list(itertools.chain.from_iterable([data_questions[questions[t]] for t in test_index]))
        with open(f'{configs.CQAEL_TRAIN_TEST_DATA}/cqa_train_data_{index}.csv', 'w') as f:
            f.write('\n'.join(train_data))
        with open(f'{configs.CQAEL_TRAIN_TEST_DATA}/cqa_test_data_{index}.csv', 'w') as f:
            f.write('\n'.join(test_data))

        train_data_metadata =  list(itertools.chain.from_iterable([data_questions_metadata[questions[t]] for t in train_index]))
        test_data_metadata = list(itertools.chain.from_iterable([data_questions_metadata[questions[t]] for t in test_index]))
        with open(f'{configs.CQAEL_TRAIN_TEST_METADATA_DATA}/cqa_train_data_{index}.csv', 'w') as f:
            f.write('\n'.join(train_data_metadata))
        with open(f'{configs.CQAEL_TRAIN_TEST_METADATA_DATA}/cqa_test_data_{index}.csv', 'w') as f:
            f.write('\n'.join(test_data_metadata))

        print(f"Data split Fold {index}: Train {len(train_data)} Test {len(test_data)}")
        index += 1


if __name__=='__main__':
    cqa_data, cqa_data_metadata = load_cqa_data(configs.CQAEL_DATASET_PATH, metadata=True)
    with open(configs.CQAEL_DATASET_ROOT_PATH+'cqa_all.csv', 'w') as result_file:
        result_file.write('\n'.join(cqa_data))
    with open(configs.CQAEL_DATASET_ROOT_PATH+'cqa_all_metadata_concat.csv', 'w') as result_file:
        result_file.write('\n'.join(cqa_data_metadata))
    split_train_test_data(cqa_data, cqa_data_metadata, 5)