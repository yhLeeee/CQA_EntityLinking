import json
import os

from tqdm import tqdm

from models.common.utils import generate_text_mentions_data, load_json, get_text_similarity_topk
import models.common.conifgs as configs

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
            paralled_texts_simtopk = get_text_similarity_topk(text, texts[:index] + texts[index+1:])
            topic_texts_simtopk = get_text_similarity_topk(text, topic_texts)
            user_texts_simtopk = get_text_similarity_topk(text, user_texts[index]) if user_texts[index] != None else []
            text_metadata_dict[text] = paralled_texts_simtopk + topic_texts_simtopk + user_texts_simtopk
            text_metadata_dict[text] = [raw_str.strip() for raw_str in text_metadata_dict[text]]
    return text_metadata_dict

def data_to_link_gen(data_path, metadata=False):
    cqa_data = load_json(data_path)

    cqa_text_ment_data = generate_text_mentions_data(cqa_data, split=False, metadata=metadata)
    data_to_link = []
    gold_entities = []
    id = 0
    for i in range(len(cqa_text_ment_data['spans'])):
        if cqa_text_ment_data['has_candidates'][i] is False:
            continue
        gold_entities.append(cqa_text_ment_data['gold_entities'][i])
        text = cqa_text_ment_data['texts'][i]
        spans = cqa_text_ment_data['spans'][i][0]
        mention = text[spans[0]:spans[0]+spans[1]]
        context_left = text[:spans[0]]
        context_right = text[spans[0]+spans[1]:]
        data_to_link.append(
            {
                'id': id,
                'label': "unknown",
                'label_id': -1,
                'context_left': context_left,
                'mention': mention,
                'context_right': context_right
            }
        )
        id += 1
    return data_to_link, gold_entities


if __name__ == '__main__':
    data_to_link_gen(configs.CQAEL_DATASET_PATH)