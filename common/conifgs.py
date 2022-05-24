PROXIES = {
    'http': 'http://127.0.0.1:7980',
    'https': 'http://127.0.0.1:7890',
}

DATASET_ROOT_PATH = '../../dataset/'

CQAEL_DATASET_ROOT_PATH = DATASET_ROOT_PATH + 'cqa-el/'

CQAEL_DATASET = 'CQAEL_dataset_20211216.json'

CQAEL_DATASET_PATH = CQAEL_DATASET_ROOT_PATH + CQAEL_DATASET

CQAEL_EMBEDDING = '0330_e180_ent2vec.txt'

CQAEL_EMBEDDING_PATH = CQAEL_DATASET_ROOT_PATH + CQAEL_EMBEDDING

CQAEL_TRAIN_TEST_DATA = CQAEL_DATASET_ROOT_PATH + 'cqa_train_test_data/'
CQAEL_TRAIN_TEST_METADATA_DATA = CQAEL_DATASET_ROOT_PATH + 'cqa_train_test_metadata_data/'

CQAEL_NCEL_DATA_PATH = CQAEL_DATASET_ROOT_PATH + 'ncel_data/'


MULREL_NEL_DATASET_ROOT_PATH = '../../../dataset/mulrel-nel/'

MULREL_NEL_DIR = MULREL_NEL_DATASET_ROOT_PATH + 'generated/test_train_data/'
MULREL_NEL_CONLL_PATH = MULREL_NEL_DATASET_ROOT_PATH + 'basic_data/test_datasets/'
MULREL_NEL_PERSON_PATH = MULREL_NEL_DATASET_ROOT_PATH + 'basic_data/p_e_m_data/persons.txt'
MULREL_NEL_VOCA_EMB_DIR = MULREL_NEL_DATASET_ROOT_PATH + 'generated/embeddings/word_ent_embs/'
