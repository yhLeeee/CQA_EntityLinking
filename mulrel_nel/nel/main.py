import os

import numpy as np

import models.mulrel_nel.nel.dataset as D
from models.mulrel_nel.nel.mulrel_ranker import MulRelRanker
from models.mulrel_nel.nel.ed_ranker import EDRanker

import models.mulrel_nel.nel.utils as utils

from pprint import pprint

import models.common.conifgs as configs

import argparse

parser = argparse.ArgumentParser()


datadir = '../' + configs.CQAEL_TRAIN_TEST_DATA
conll_path = '../' + configs.CQAEL_TRAIN_TEST_DATA
person_path = configs.MULREL_NEL_PERSON_PATH
voca_emb_dir = configs.MULREL_NEL_VOCA_EMB_DIR
# cqa_entity_voca_emb_dir = '../' + configs.CQAEL_DATASET_ROOT_PATH

ModelClass = MulRelRanker

parser.add_argument("--metadata", action='store_true', help="use metadata dataset")
parser.add_argument("--deeped", action='store_true', help="use deeped model")

# general args
parser.add_argument("--cuda_device", type=str,
                    default="0", help="which gpu to use")

parser.add_argument("--mode", type=str,
                    help="train or eval",
                    default='train')
parser.add_argument("--model_path", type=str,
                    help="model path to save/load",
                    default='')

parser.add_argument("--cqa_entity_voca_emb_dir", type=str,
                    help="CQA entity voca emb path to save/load",
                    default='../' + configs.CQAEL_DATASET_ROOT_PATH)

# args for preranking (i.e. 2-step candidate selection)
parser.add_argument("--n_cands_before_rank", type=int,
                    help="number of candidates",
                    default=20)
parser.add_argument("--prerank_ctx_window", type=int,
                    help="size of context window for the preranking model",
                    default=50)
parser.add_argument("--keep_p_e_m", type=int,
                    help="number of top candidates to keep w.r.t p(e|m)",
                    default=15)
parser.add_argument("--keep_ctx_ent", type=int,
                    help="number of top candidates to keep w.r.t using context",
                    default=5)

# args for local model
parser.add_argument("--ctx_window", type=int,
                    help="size of context window for the local model",
                    default=100)
parser.add_argument("--tok_top_n", type=int,
                    help="number of top contextual words for the local model",
                    default=25)


# args for global model
parser.add_argument("--mulrel_type", type=str,
                    help="type for multi relation (rel-norm or ment-norm)",
                    default='ment-norm')
parser.add_argument("--n_rels", type=int,
                    help="number of relations",
                    default=5)
parser.add_argument("--hid_dims", type=int,
                    help="number of hidden neurons",
                    default=100)
parser.add_argument("--snd_local_ctx_window", type=int,
                    help="local ctx window size for relation scores",
                    default=6)
parser.add_argument("--dropout_rate", type=float,
                    help="dropout rate for relation scores",
                    default=0.3)


# args for training
parser.add_argument("--n_epochs", type=int,
                    help="max number of epochs",
                    default=200)
parser.add_argument("--dev_f1_change_lr", type=float,
                    help="dev f1 to change learning rate",
                    default=0.85)
parser.add_argument("--n_not_inc", type=int,
                    help="number of evals after dev f1 not increase",
                    default=10)
parser.add_argument("--eval_after_n_epochs", type=int,
                    help="number of epochs to eval",
                    default=5)
parser.add_argument("--learning_rate", type=float,
                    help="learning rate",
                    default=1e-4)
parser.add_argument("--margin", type=float,
                    help="margin",
                    default=0.01)

# args for LBP
parser.add_argument("--df", type=float,
                    help="dumpling factor (for LBP)",
                    default=0.5)
parser.add_argument("--n_loops", type=int,
                    help="number of LBP loops",
                    default=10)

# args for debugging
parser.add_argument("--print_rel", action='store_true')
parser.add_argument("--print_incorrect", action='store_true')


args = parser.parse_args()


if __name__ == "__main__":
    print(args)

    # set cuda device
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device

    if args.metadata:
        datadir = '../' + configs.CQAEL_TRAIN_TEST_METADATA_DATA
        conll_path = '../' + configs.CQAEL_TRAIN_TEST_METADATA_DATA

    cqa_entity_voca_emb_dir = args.cqa_entity_voca_emb_dir

    print('load conll at', datadir)
    conll = D.CoNLLDataset(datadir, person_path, conll_path)

    train_test_datasets = [(conll.train_cqa_0, ('CQA-test-0', conll.test_cqa_0)),
                           (conll.train_cqa_1, ('CQA-test-1', conll.test_cqa_1)),
                           (conll.train_cqa_2, ('CQA-test-2', conll.test_cqa_2)),
                           (conll.train_cqa_3, ('CQA-test-3', conll.test_cqa_3)),
                           (conll.train_cqa_4, ('CQA-test-4', conll.test_cqa_4))]

    f1_scores = []
    for i, train_test_data in enumerate(train_test_datasets):

        print('create model')
        word_voca, word_embeddings = utils.load_voca_embs(voca_emb_dir + 'dict.word',
                                                          voca_emb_dir + 'word_embeddings.npy')
        print('word voca size', word_voca.size())
        snd_word_voca, snd_word_embeddings = utils.load_voca_embs(voca_emb_dir + '/glove/dict.word',
                                                                  voca_emb_dir + '/glove/word_embeddings.npy')
        print('snd word voca size', snd_word_voca.size())

        # entity_voca, entity_embeddings = utils.load_voca_embs(cqa_entity_voca_emb_dir + 'dict.entity',
        #                                                       cqa_entity_voca_emb_dir + 'entity_embeddings.npy')

        entity_voca, entity_embeddings = utils.load_voca_embs(cqa_entity_voca_emb_dir + '/cqa_dict.entity',
                                                              cqa_entity_voca_emb_dir + '/cqa_entity_embeddings.npy')

        config = {'hid_dims': args.hid_dims,
                  'emb_dims': entity_embeddings.shape[1],
                  'freeze_embs': True,
                  'tok_top_n': args.tok_top_n,
                  'margin': args.margin,
                  'word_voca': word_voca,
                  'entity_voca': entity_voca,
                  'word_embeddings': word_embeddings,
                  'entity_embeddings': entity_embeddings,
                  'snd_word_voca': snd_word_voca,
                  'snd_word_embeddings': snd_word_embeddings,
                  'dr': args.dropout_rate,
                  'args': args,
                  'deeped': args.deeped}

        if ModelClass == MulRelRanker:
            config['df'] = args.df
            config['n_loops'] = args.n_loops
            config['n_rels'] = args.n_rels
            config['mulrel_type'] = args.mulrel_type
        else:
            raise Exception('unknown model class')

        pprint(config)
        ranker = EDRanker(config=config)

        if args.mode == 'train':
            print('training...')
            config = {'lr': args.learning_rate, 'n_epochs': args.n_epochs}
            pprint(config)
            # ranker.train(conll.train_cqa_1, dev_datasets, config)



            print('############################################################')
            print(f'Training on train dataset {i}')
            print('############################################################')
            f1_temp = ranker.train(train_test_data[0], [train_test_data[1],], config)
            f1_scores.append(f1_temp)
    print('############################################################')
    print(f'Average F1 score: {np.mean(f1_scores)}')
    print('############################################################')
