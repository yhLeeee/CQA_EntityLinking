import models.BLINK.blink.main_dense as main_dense
import argparse
from sklearn.metrics import accuracy_score
import os
os.environ['http_proxy'] = 'http://127.0.0.1:7890'
os.environ['https_proxy'] = 'http://127.0.0.1:7890'


from models.BLINK.blink_data_generate import data_to_link_gen
import models.common.conifgs as configs

models_path = "./models/" # the path where you stored the BLINK models

config = {
    "test_entities": None,
    "test_mentions": None,
    "interactive": False,
    "top_k": 10,
    "biencoder_model": models_path+"biencoder_wiki_large.bin",
    "biencoder_config": models_path+"biencoder_wiki_large.json",
    "entity_catalogue": models_path+"entity.jsonl",
    "entity_encoding": models_path+"all_entities_large.t7",
    "crossencoder_model": models_path+"crossencoder_wiki_large.bin",
    "crossencoder_config": models_path+"crossencoder_wiki_large.json",
    "fast": False, # set this to be true if speed is a concern
    "output_path": "../../logs/" # logging directory
}

args = argparse.Namespace(**config)

models = main_dense.load_models(args, logger=None)

data_to_link, gold_entities = data_to_link_gen(configs.CQAEL_DATASET_PATH, metadata=True)

predictions = [0] * len(gold_entities)
k = 2
for i in range(k):
    m = int(len(data_to_link) / k)
    data = data_to_link[i*m:(i+1)*m]
    ids = [x['id'] for x in data]
    renumber = 0
    for j in range(len(data)):
        data[j]['id'] = renumber
        renumber += 1
    _, _, _, _, _, preds, scores, = main_dense.run(args, None, *models, test_data=data)
    for j in range(len(preds)):
        predictions[ids[j]] = preds[j][0]
print()
print(predictions)
print()
print(gold_entities)
acc = accuracy_score(gold_entities, predictions)
print()
print("Accuracy:", acc)