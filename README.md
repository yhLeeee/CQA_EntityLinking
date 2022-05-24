# CQA_EntityLinking

Codes and data set for IJCAI2022: Community Question Answering Entity Linking via Leveraging Auxiliary Data.

## Requirements

- torch == 1.8.0+
- transformers == 4.5.1

## Dataset: QuoraEL

We construct a new dataset **QuoraEL**, which contains data of 504 CQA texts in total. The Wikipedia dump (July 2019 version) is used as the reference KB. Since our data set folder is too large, we release it [here](https://drive.google.com/drive/folders/1dW6iw268uDbBdi7opfwOAz_zFyq7DlrH). Our data are in the folder data sets. CQAEL_dataset.json contains QuaraEL data mentioned above. Details of other files can be found in the codes for format conversion.
### Data format

1. For each **question**, the following items are covered:`question title`, `question url`, `ID of question`, `answers`, `mentions in question title`, `topics` .

   `topics` includes `topic name`, `topic url`, `questions under this topic`

2. For each **answer**, the following items are covered:

   `answer url`, `answer id`, `upvote count`, `answer content`, `mentions in answer content`, `user name`, `user url`, `user history answers`, `user history questions`

3. For each **mention**, the following items are covered:

   `mention text`, `corresponding entity`, `candidates`, `gold entity index`

   `candidates` is a string and each candidate in `Candidates` is like: 

    `<ENTITY>\t<WIKIPEDIA_ID>\t<PRIOR_PROB>`

   The index of gold entity is '-1' if the mention cannot be linked to any candidates. There are 8030 mentions that can be linked to some candidate.

### Load data

The data set is constructed in **json** format. You can load it easily.

```python
import json
with open(PATH_OF_DATASET_FILE, 'r') as fp:
  data = json.load(fp)
```



## Files

- `models` folder: Codes of our model and baseline models. Baselines includes [Deep-ED](https://github.com/dalab/deep-ed), [Ment-Norm](https://github.com/lephong/mulrel-nel), [FGS2EE](https://github.com/fhou80/EntEmb), [Zeshel](https://github.com/lajanugen/zeshel), [REL](https://github.com/informagi/REL), [BLINK](https://github.com/facebookresearch/BLINK), [GENRE](https://github.com/facebookresearch/GENRE). Some data files can be downloaded via links in their original repository. 
- `dataset` folder: our data are in the subfolder `cqa-el`. `CQAEL_dataset.json` contains **QuaraEL** data mentioned above. Details of other files can be found in the codes for format conversion.



For more details about the data set and the experiment settings, please refer to our paper.
