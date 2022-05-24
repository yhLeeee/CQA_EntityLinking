import itertools
import json
import time

from tqdm import tqdm
import requests
from sklearn.metrics import accuracy_score
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
from models.common.utils import load_json, generate_text_mentions_data
import models.common.conifgs as configs

def rel_request(text, span):
    API_URL = "https://rel.cs.ru.nl/api"

    s = requests.session()
    retry = Retry(connect=5, backoff_factor=0.75)
    adapter = HTTPAdapter(max_retries=retry)
    s.mount('http://', adapter)
    s.mount('https://', adapter)
    s.keep_alive = False  # 关闭多余连接

    el_result = s.post(API_URL,
                       proxies=configs.PROXIES,
                       json={
                           "text": text,
                           "spans": span
                       }).json()
    return el_result

def rel(data, metadata=False):
    """
    :param data: dict
    :return: list
    """
    texts = data['texts']
    spans = data['spans']
    linked_entities = []
    for i in tqdm(range(len(texts))):
        try_times = 5
        while try_times>=0:
            try:
                el_result = rel_request(texts[i], spans[i])
                break
            except Exception as e:
                print(e)
                print(el_result, i, texts[i], spans[i])
                el_result = []
                try_times -= 1
                print("Sleep for 3 seconds. Try again... Left Times:", try_times)
                time.sleep(3)
        linked_result = [res[3] for res in el_result]
        linked_entities.append(linked_result)
    data['rel_results'] = linked_entities
    if metadata:
        save_name = './rel_result_metadata.json'
    else:
        save_name = './rel_result.json'
    with open(save_name, 'w') as f:
        json.dump(data, f)
    return linked_entities


if __name__ == '__main__':
    text = "The quick brown fox jumps over the lazy dog."
    span = [(16, 3), (40, 3)]
    res = rel_request(text, span)
    print(res)

    cqa_data = load_json(configs.CQAEL_DATASET_PATH)
    cqa_text_ment_data = generate_text_mentions_data(cqa_data, split=False, metadata=True)
    mentions_count = len(cqa_text_ment_data['gold_entities'])

    # linked_entities = rel(cqa_text_ment_data)
    # linked_data = load_json('./rel_result.json')

    # linked_entities = rel(cqa_text_ment_data, metadata=True)
    linked_data = load_json('./rel_result_metadata.json')
    # assert len(linked_data['rel_results']) == len(cqa_text_ment_data['gold_entities'])

    gold_entities_unpack = cqa_text_ment_data['gold_entities']
    linked_entities_unpack = []
    for e in linked_data['rel_results']:
        if len(e)!= 0:
            assert len(e) == 1
            linked_entities_unpack.append(e[0].replace('_', ' '))
        else:
            linked_entities_unpack.append('')
    # linked_entities_unpack = list(itertools.chain.from_iterable(linked_entities))
    error = 0
    count = 0
    gold_entities = []
    linked_entities = []
    for i in range(len(gold_entities_unpack)):
        assert cqa_text_ment_data['texts'][i] == linked_data['texts'][i]
        assert cqa_text_ment_data['spans'][i][0][0] == linked_data['spans'][i][0][0]
        assert cqa_text_ment_data['spans'][i][0][1] == linked_data['spans'][i][0][1]
        assert cqa_text_ment_data['gold_entities'][i] == linked_data['gold_entities'][i]
        if cqa_text_ment_data['has_candidates'][i] is False:
            continue
        count += 1
        gold_entities.append(gold_entities_unpack[i])
        linked_entities.append(linked_entities_unpack[i])
        if gold_entities_unpack[i] != linked_entities_unpack[i]:
            error += 1
            print(i, gold_entities_unpack[i], '|', linked_entities_unpack[i])
    acc = accuracy_score(gold_entities, linked_entities)
    print("Accuracy:", acc)
    print("All:", count)
    print("Error:", error)