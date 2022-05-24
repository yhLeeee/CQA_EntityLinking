from models.common.utils import load_json, generate_text_mentions_data
import models.common.conifgs as configs


def data_to_link_gen(data_path, metadata=False):
    cqa_data = load_json(data_path)

    cand = {}
    for qa in cqa_data['questions']:
        for m in qa['mentions']:
            cand[m['mention']]= m['Candidates'].split('\t')[0::3]
        for ans in qa['answers']:
            for m in ans['mentions']:
                cand[m['mention']] = m['Candidates'].split('\t')[0::3]

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
                'context_right': context_right,
                'candidates': cand[mention] if mention in cand else [],
            }
        )
        id += 1
    return data_to_link, gold_entities


if __name__ == '__main__':
    data_to_link_gen(configs.CQAEL_DATASET_PATH)