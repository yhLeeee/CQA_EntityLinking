# REL: Radboud Entity Linker

![API status](https://img.shields.io/endpoint?label=status&url=https%3A%2F%2Frel.cs.ru.nl%2Fapi)
![build](https://github.com/informagi/REL/workflows/build/badge.svg)

REL is a modular Entity Linking package that is provided as a Python package as well as a web API. REL has various meanings -  one might first notice that it stands for relation, which is a suiting name for the problems that can be tackled with this package. Additionally, in Dutch a 'rel' means a disturbance of the public order, which is exactly what we aim to achieve with the release of this package.

REL utilizes *English* Wikipedia as a knowledge base and can be used for the following tasks:
- **Entity linking (EL)**: Given a text, the system outputs a list of mention-entity pairs, where each mention is a n-gram from text and each entity is an entity in the knowledge base.
- **Entity Disambiguation (ED)**: Given a text and a list of mentions, the system assigns an entity (or NIL) to each mention.

# REL variants

REL comes in two variants for identifying entity mentions:

- **Case-sensitive**: This setup is suitable for properly written texts (e.g., news articles) and is the default setup of the REL package. In this setup, we use the `ner-fast` FLAIR model, which is case-sensitive. The results reported in the REL paper are based on this model.

- **Case-insensitive**: This setup is well suited for noisy texts (e.g., queries), where entity mentions can be (often) lowercased. In this setup, we use the `ner-fast-with-lowercase` model, which is the `ner-fast` FLAIR architucture trained on randomly cased and uncased text. This variant is the default setup of our API.

Below is a comparison of these two models on [CoNLL-2003 NER](https://www.clips.uantwerpen.be/conll2003/ner/) dataset.

| Model  | CoNLL-2003 test | F1 |
| ------ | --------------- | -- |
| `ner-fast`  |  original | 92.78 |
| `ner-fast`  |  lower-cased | 58.42 |
| `ner-fast`  |  random | 70.64 |
| `ner-fast-with-lowercase`  |  original | 91.53|
| `ner-fast-with-lowercase`  |  lower-cased | 89.73 |
| `ner-fast-with-lowercase`  |  random | 89.66 |

See [Notes on using custom models](https://github.com/informagi/REL/tree/master/tutorials/07_custom_models.md) for further information on switiching between these variants.


# Calling our API

Users may access our API by using the example script below. 
For EL, the `spans` field needs to be set to an empty list. For ED, however, the `spans` field should consist of a list of tuples, where each tuple refers to the start position and length of a mention.

```python
import requests

API_URL = "https://rel.cs.ru.nl/api"
text_doc = "If you're going to try, go all the way - Charles Bukowski"

# Example EL.
el_result = requests.post(API_URL, json={
    "text": text_doc,
    "spans": []
}).json()

# Example ED.
ed_result = requests.post(API_URL, json={
    "text": text_doc,
    "spans": [(41, 16)]
}).json()
```

# Setup package
This section describes how to deploy REL on a local machine and setup the API. If you want to do anything more than simply running our API locally, you can skip the Docker steps and continue with installation from source.

## Option 1: Installation using Docker
First, download the necessary data; you need the generic files and a Wikipedia version (2014 or 2019) (see [Download](#download)). Extract them anywhere, we will bind the directories to the Docker container as volumes.

```bash
./scripts/download_data.sh ./data generic wiki_2019
```

### Prebuilt images
To use our prebuilt default image, run:
```bash
docker pull informagi/rel
```

To run the API locally:
```bash
# Map container port 5555 to local port 5555, and use Wikipedia 2019
# Also map the generic and wiki_2019 folders to directories in Docker container
docker run \
    -p 5555:5555 \
    -v $PWD/data/:/workspace/data \
    --rm -it informagi/rel \
    python -m REL.server --bind 0.0.0.0 --port 5555 /workspace/data wiki_2019
```

Now you can make requests to `http://localhost:5555` (or another port if you
use a different mapping) in the format described in the example above.

### Build your own Docker image
To build the Docker image yourself, run:
```bash
# Clone the repository
git clone https://github.com/informagi/REL && cd REL
# Build the Docker image
docker build . -t informagi/rel
```

To run the API locally, use the same commands as mentioned in the previous section.

##  Option 2: Installation from source code
Run the following command in a terminal to install REL:
```
pip install git+https://github.com/informagi/REL
```
You will also need to manually download the files described in the next section.

## Download
The files used for this project can be divided into three categories. The first is a generic set of documents and embeddings that was used throughout the project. This folder includes the GloVe embeddings and the unprocessed datasets that were used to train the ED model. The second and third category are Wikipedia corpus related files, which in our case either originate from a 2014 or 2019 corpus. Alternatively, users may use their own corpus, for which we refer to the tutorials.

* [Download generic files](http://gem.cs.ru.nl/generic.tar.gz)
* [Download Wikipedia corpus (2014)](http://gem.cs.ru.nl/wiki_2014.tar.gz)
* [Download ED model 2014](http://gem.cs.ru.nl/ed-wiki-2014.tar.gz)
* [Download Wikipedia corpus (2019)](http://gem.cs.ru.nl/wiki_2019.tar.gz)
* [Download ED model 2019](http://gem.cs.ru.nl/ed-wiki-2019.tar.gz)

## Tutorials
To promote usage of this package we developed various tutorials. If you simply want to use our API, then 
we refer to the section above. If you feel one is missing or unclear, then please create an issue, which is much appreciated :)! The first two tutorials are
for users who simply want to use our package for EL/ED and will be using the data files that we provide. 
The remainder of the tutorials are optional and for users who wish to e.g. train their own Embeddings.

1. [How to get started (project folder and structure).](https://github.com/informagi/REL/tree/master/tutorials/01_How_to_get_started.md)
2. [End-to-End Entity Linking.](https://github.com/informagi/REL/tree/master/tutorials/02_E2E_Entity_Linking.md)
3. [Evaluate on GERBIL.](https://github.com/informagi/REL/tree/master/tutorials/03_Evaluate_Gerbil.md)
4. [Deploy REL for a new Wikipedia corpus](https://github.com/informagi/REL/tree/master/tutorials/deploy_REL_new_Wiki/04_deploy_REL_new_wiki.md):
    1. [Extracting a new Wikipedia corpus and creating a p(e|m) index.](https://github.com/informagi/REL/tree/master/tutorials/deploy_REL_new_Wiki/04_01_Extracting_a_new_Wikipedia_corpus.md)
    2. [Training your own Embeddings.](https://github.com/informagi/REL/tree/master/tutorials/deploy_REL_new_Wiki/04_02_training_your_own_embeddings.md)
    3. [Generating training, validation and test files.](https://github.com/informagi/REL/tree/master/tutorials/deploy_REL_new_Wiki/04_03_generating_training_test_files.md)
    4. [Training your own Entity Disambiguation model.](https://github.com/informagi/REL/tree/master/tutorials/deploy_REL_new_Wiki/04_04_training_your_own_ED_model.md)
5. [Reproducing our results](https://github.com/informagi/REL/tree/master/tutorials/05_reproducing_our_results.md)
6. [REL as systemd service](https://github.com/informagi/REL/tree/master/tutorials/06_systemd_instructions.md)
7. [Notes on using custom models](https://github.com/informagi/REL/tree/master/tutorials/07_custom_models.md)

# Cite
If you are using REL, please cite the following paper:

```bibtex
@inproceedings{vanHulst:2020:REL,
 author =    {van Hulst, Johannes M. and Hasibi, Faegheh and Dercksen, Koen and Balog, Krisztian and de Vries, Arjen P.},
 title =     {REL: An Entity Linker Standing on the Shoulders of Giants},
 booktitle = {Proceedings of the 43rd International ACM SIGIR Conference on Research and Development in Information Retrieval},
 series =    {SIGIR '20},
 year =      {2020},
 publisher = {ACM}
}
```

# Contact
If you find any bugs or experience difficulties when using REL, please create a issue on this Github page. If you have any specific questions with respect to our research with REL, please email [Mick van Hulst](mailto:mick.vanhulst@gmail.com).

# Acknowledgements
Our thanks go out to the authors that open-sourced their code, enabling us to create this package that can hopefully be of service to many.
