import json
import os

import Levenshtein
import difflib

from tqdm import tqdm

import models.common.conifgs as configs


def load_json(filename):
    with open(filename, 'r') as f:
        return json.load(f)


def get_mention_pos(text, ment):
    start = text.find(ment)
    if (text[start:start + len(ment)] != ment):
        print("Error: mention not found")
        print(" - ", text)
        print(" - ", ment)
        start = len(text) - len(ment)
    return start, len(ment)


def get_text_similarity_topk(text, to_compare, topk=3):
    sims = []
    for t in to_compare:
        s1 = difflib.SequenceMatcher(text, t).ratio()
        s2 = Levenshtein.ratio(text, t)
        # s3 = edit_distance.SequenceMatcher(text, t).ratio()
        # sims.append((s1+s2+s3)/3)
        sims.append((s1 + s2) / 2)
    indices = sorted(range(len(sims)), key=lambda i: sims[i], reverse=True)[:topk]
    return [to_compare[i] for i in indices]


def add_metadata(raw_data, K):
    '''
    Meta data format:
    [context_right]
    ==>
    [context_right] [concatation of parallel answers] [concatation of topic name and questions] [*concatation of user questions or answers]

    return:
      a list of meta data strings
    '''

    text_metadata_dict = {}
    raw_data = raw_data['questions']
    for q in tqdm(raw_data):
        topic_texts = [' '.join((t['topic_name'], tq)) for t in q['topics'] for tq in t['topic_question']]
        texts = []
        user_texts = []

        texts.append(q['question_title'])
        user_texts.append(None)

        for ans in q['answers']:
            texts.append(ans['answer_content'])
            user_texts.append(ans['user_question'] + ans['user_answer'])

        for index, text in enumerate(texts):
            paralled_texts_simtopk = get_text_similarity_topk(text, texts[:index] + texts[index + 1:])
            topic_texts_simtopk = get_text_similarity_topk(text, topic_texts)
            user_texts_simtopk = get_text_similarity_topk(text, user_texts[index]) if user_texts[index] != None else []
            text_metadata_dict[text] = paralled_texts_simtopk + topic_texts_simtopk + user_texts_simtopk
            text_metadata_dict[text] = [raw_str.strip() for raw_str in text_metadata_dict[text]]
    return text_metadata_dict


def generate_text_mentions_data(cqa_data, split=False, metadata=False):
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
    for question in questions:
        texts = []
        spans = []
        gold_entities = []
        has_candidates = []

        ###########################
        #     process question    #
        ###########################

        q_text = question['question_title']
        q_ments = question['mentions']

        if metadata:
            texts.append(q_text + ' ' + ' '.join(metadata_dict[q_text]))
        else:
            texts.append(q_text)

        tmp_span_list = []
        tmp_entity_list = []
        tmp_has_candidate = []
        for q_ment in q_ments:
            q_ment_start, q_ment_len = get_mention_pos(q_text, q_ment['mention'])
            gold_entity = q_ment['entity']
            cand = q_ment['Gold_index']
            tmp_span_list.append((q_ment_start, q_ment_len))
            tmp_entity_list.append(gold_entity)
            tmp_has_candidate.append(cand != -1)
        spans.append(tmp_span_list)
        gold_entities.append(tmp_entity_list)
        has_candidates.append(tmp_has_candidate)

        ###########################
        #     process answers     #
        ###########################
        for answer in question['answers']:
            a_text = answer['answer_content']
            a_ments = answer['mentions']

            if metadata:
                texts.append(a_text + ' ' + ' '.join(metadata_dict[a_text]))
            else:
                texts.append(a_text)

            tmp_span_list = []
            tmp_entity_list = []
            tmp_has_candidate = []
            for a_ment in a_ments:
                a_ment_start, a_ment_len = get_mention_pos(a_text, a_ment['mention'])
                gold_entity = a_ment['entity']
                cand = a_ment['Gold_index']
                tmp_span_list.append((a_ment_start, a_ment_len))
                tmp_entity_list.append(gold_entity)
                tmp_has_candidate.append(cand != -1)
            spans.append(tmp_span_list)
            gold_entities.append(tmp_entity_list)
            has_candidates.append(tmp_has_candidate)

        ret_data.append({
            'texts': texts,
            'spans': spans,
            'gold_entities': gold_entities,
            'has_candidates': has_candidates
        })
    if not split:
        combined_data = {
            'texts': [],
            'spans': [],
            'gold_entities': [],
            'has_candidates': []
        }
        for data in ret_data:
            assert len(data['gold_entities']) == len(data['spans'])
            for i in range(len(data['spans'])):
                assert len(data['spans'][i]) == len(data['gold_entities'][i])
                combined_data['texts'] += [data['texts'][i], ] * len(data['spans'][i])
                combined_data['spans'] += data['spans'][i]
                combined_data['gold_entities'] += data['gold_entities'][i]
                combined_data['has_candidates'] += data['has_candidates'][i]
        combined_data['spans'] = [[t] for t in combined_data['spans']]
        return combined_data
    return ret_data
