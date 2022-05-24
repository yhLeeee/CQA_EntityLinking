import os
import pickle

from sklearn.metrics import accuracy_score

os.environ['http_proxy'] = 'http://127.0.0.1:7890'
os.environ['https_proxy'] = 'http://127.0.0.1:7890'

from genre.trie import Trie
# for pytorch/fairseq
from genre.fairseq_model import GENRE

import models.common.conifgs as configs
from genre_data_to_link import data_to_link_gen
from genre.entity_linking import get_end_to_end_prefix_allowed_tokens_fn_fairseq as get_prefix_allowed_tokens_fn

data_to_link, gold_entities = data_to_link_gen(configs.CQAEL_DATASET_PATH, metadata=True)
model = GENRE.from_pretrained("./models/fairseq_entity_disambiguation_aidayago").eval()
# load the prefix tree (trie)
with open("./data/kilt_titles_trie_dict.pkl", "rb") as f:
    trie = Trie.load_from_dict(pickle.load(f))

n = 1000
predictions = []
for i in range(0, len(data_to_link), n):
    print(i)
    sentences = [f"{data['context_left']} [START_ENT] {data['mention']} [END_ENT] {data['context_right']}" for data in data_to_link[i:i+n]]

    model_output = model.sample(
        sentences,
        prefix_allowed_tokens_fn=lambda batch_id, sent: trie.get(sent.tolist()),
        # prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
    )


    def get_predict_entity(list_of_dict):
        res = sorted(list_of_dict, key=lambda d: d['score'], reverse=True)
        return res[0]['text']


    predictions_sub = [get_predict_entity(d) for d in model_output]
    predictions.extend(predictions_sub)

for i in range(len(predictions)):
    print(gold_entities[i], '\t', predictions[i])

acc = accuracy_score(gold_entities, predictions)
print()
print("Accuracy:", acc)
