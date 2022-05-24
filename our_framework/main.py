import argparse
import json
import itertools
from torch.autograd.grad_mode import F
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from transformers.utils.dummy_pt_objects import Trainer
from tools import *
from DataProcess import *
import torch
import torch.nn as nn
import CQA_Model
from torch.utils.data import DataLoader
from pytorch_transformers.optimization import AdamW, WarmupLinearSchedule
import random
import numpy as np
from tqdm import tqdm, trange
import tools
import torch.distributed as dist
import torch.utils.data.distributed
from torch.nn import DataParallel

device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')

def getalldata(datafile):
    f = open(datafile, 'r', encoding='utf-8')
    qa = []
    qa_index = []
    qa_json = json.load(f)
    for q_index, q in enumerate(qa_json["questions"]):
        qa.append(q)
        qa_index.append(q_index)
    return list(itertools.zip_longest(qa, qa_index))


def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train(model, train_dataloader, optimizer, scheduler, epoch_idx, logger, args, LEN_TRAIN_SET):
    model.train()
    tr_total_loss = 0
    train_accuarcy = 0
    total_train_examples = 0

    # for step, batch in enumerate(tqdm(train_dataloader, desc="Batch")):
    for step, batch in enumerate(train_dataloader):
        batch = tuple(t.cuda()
                      for t in batch)
        all_ctxt_cross_ids, all_ctxt_cross_mask, all_ctxt_cross_seg, \
        all_cqa_cross_ids, \
        all_candidate_priors, label_ids, all_entity_masks, all_topic_cross_ids, all_user_cross_ids, all_uesr_flag, = batch
        loss, logits = model(
            all_ctxt_cross_ids,
            all_ctxt_cross_mask,
            all_ctxt_cross_seg,
            all_cqa_cross_ids,
            all_topic_cross_ids,
            all_user_cross_ids, 
            all_uesr_flag,
            all_candidate_priors,
            label_ids,
            all_entity_masks,
        )

        loss = loss.sum()

        if args.gradient_accumulation_steps > 1:
            loss = loss / args.gradient_accumulation_steps
        loss.backward()

        if (step + 1) % (
            args.print_tr_loss_opt_steps_interval
            * args.gradient_accumulation_steps
        ) == 0:
            logger.info(
                'Epoch [{:d}/{:d}], Iter[{:d}/{:d}], batch_train_loss:{:.9f}'\
                .format(epoch_idx + 1, args.epochs, step, len(train_dataloader), loss.item())
            )

        if (step + 1) % args.gradient_accumulation_steps == 0:
            nn.utils.clip_grad_norm_(
                model.parameters(), args.max_grad_norm
            )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        tr_total_loss += loss.item()

        logits = logits.detach().cpu().numpy()
        label_ids = label_ids.to("cpu").numpy()
        temp_train_accuarcy = tools.accuracy(logits, label_ids)

        train_accuarcy += temp_train_accuarcy
        total_train_examples += all_ctxt_cross_ids.size(0)

    normalized_train_accuarcy = train_accuarcy/LEN_TRAIN_SET

    return tr_total_loss/total_train_examples, normalized_train_accuarcy


def eval(model, vali_dataloader, args, LEN_EVAL_SET):
    model.eval()
    eval_accuarcy = 0

    for all_ctxt_cross_ids, all_ctxt_cross_mask, all_ctxt_cross_seg, \
        all_cqa_cross_ids, \
        all_candidate_priors, label_ids, all_entity_masks, all_topic_cross_ids, all_user_cross_ids, all_uesr_flag, in vali_dataloader:

        all_ctxt_cross_ids = all_ctxt_cross_ids.cuda()
        all_ctxt_cross_mask = all_ctxt_cross_mask.cuda()
        all_ctxt_cross_seg = all_ctxt_cross_seg.cuda()
        all_cqa_cross_ids = all_cqa_cross_ids.cuda()
        all_topic_cross_ids = all_topic_cross_ids.cuda()
        all_user_cross_ids = all_user_cross_ids.cuda()
        all_uesr_flag = all_uesr_flag.cuda()
        all_candidate_priors = all_candidate_priors.cuda()
        label_ids = label_ids.cuda()
        all_entity_masks = all_entity_masks.cuda()

        with torch.no_grad():
            _, logits = model(
                all_ctxt_cross_ids,
                all_ctxt_cross_mask,
                all_ctxt_cross_seg,
                all_cqa_cross_ids,
                all_topic_cross_ids,
                all_user_cross_ids,
                all_uesr_flag,
                all_candidate_priors,
                label_ids,
                all_entity_masks,
            )

        logits = logits.detach().cpu().numpy()
        label_ids = label_ids.to("cpu").numpy()
        temp_eval_accuarcy = tools.accuracy(logits, label_ids)

        eval_accuarcy += temp_eval_accuarcy

    normalized_eval_accuarcy = eval_accuarcy/LEN_EVAL_SET

    return normalized_eval_accuarcy


def main(args):
    logger = tools.logger_config(log_path='log.txt', logging_name='lyh')
    logger.info(
        "*" * 32 + "Community Question Answering Entity Linking Leveraging Auxiliary Data" + "*" * 32)

    alldata = getalldata(args.data_file)

    LEN_TRAIN_SET = 0
    LEN_EVAL_SET = 0
    LEN_TEST_SET = 0

    if not args.vali:
        train_set, test_set = BuildDataSet.build_train_test(
            alldata, args.split_index)
        logger.info('The train_set size is {:}'.format(len(train_set)))
        logger.info('The test_set size is {:}'.format(len(test_set)))
        LEN_TRAIN_SET = len(train_set)
        LEN_TEST_SET = len(test_set)
    else:
        train_set, vali_set, test_set = BuildDataSet.build_train_vali_test_all(
            alldata, args.split_index, args.top_k)
        logger.info('The train_set size of split_index {} is {:}'.format(args.split_index, len(train_set)))
        logger.info('The vali_set size of split_index {} is {:}'.format(args.split_index, len(vali_set)))
        logger.info('The test_set size of split_index {} is {:}'.format(args.split_index, len(test_set)))
        LEN_TRAIN_SET = len(train_set)
        LEN_EVAL_SET = len(vali_set)
        LEN_TEST_SET = len(test_set)

    if args.train_and_eval:
        logger.info('Initing model to train....')
    else:
        logger.info('Loading existing model....')

    model = CQA_Model.CQAEL(args)

    if args.cuda:
        model = model.cuda()
    model = DataParallel(model,
                         device_ids=[0, 1],
                         output_device=device)

    set_seed(args.seed)

    xlnet_tokenizer = model.module.xlnet_tokenizer
    xlnet_tokenizer.add_special_tokens(
        {'additional_special_tokens': ["[ENT]"]})
    model.module.c_encoder.resize_token_embeddings(len(xlnet_tokenizer))
    longformer_tokenizer = model.module.longformer_tokenizer

    # store_path
    result_path = args.model_storage_path
    PATH = os.path.join(
        result_path, 'net_split_{}'.format(args.split_index))

    if args.train_and_eval == True:
        logger.info("Loading training data...")
        train_data, train_tensor_data = model.module._process_mentions_for_model(
            train_set, args.top_k, xlnet_tokenizer, longformer_tokenizer, args.max_seq_length, args.max_desc_length, args.max_q_length, args.debug, blink=args.useBLINKDic, logger=logger, 
            num_of_cqa=args.max_cqa_nums, max_user_q_nums=args.max_user_q_nums, max_topic_q_nums=args.max_topic_q_nums, max_topic_nums=args.max_topic_nums,)

        train_sampler = RandomSampler(train_tensor_data)
        train_dataloader = DataLoader(
            train_tensor_data, sampler=train_sampler, batch_size=args.batch_size, num_workers=8, pin_memory=True,
        )

        logger.info("Loading validation data...")
        vali_data, vali_tensor_data = model.module._process_mentions_for_model(
            vali_set, args.top_k, xlnet_tokenizer, longformer_tokenizer, args.max_seq_length, args.max_desc_length, args.max_q_length, args.debug, blink=args.useBLINKDic, logger=logger,
            num_of_cqa=args.max_cqa_nums, max_user_q_nums=args.max_user_q_nums, max_topic_q_nums=args.max_topic_q_nums, max_topic_nums=args.max_topic_nums,
        )
        vali_sampler = SequentialSampler(vali_tensor_data)
        vali_dataloader = DataLoader(
            vali_tensor_data, sampler=vali_sampler, batch_size=args.batch_size, num_workers=8, pin_memory=True,
        )

        test_data, test_tensor_data = model.module._process_mentions_for_model(
            test_set, args.top_k, xlnet_tokenizer, longformer_tokenizer, args.max_seq_length, args.max_desc_length, args.max_q_length, args.debug, blink=args.useBLINKDic, logger=logger,
            num_of_cqa=args.max_cqa_nums, max_user_q_nums=args.max_user_q_nums, max_topic_q_nums=args.max_topic_q_nums, max_topic_nums=args.max_topic_nums,
        )
        test_sampler = SequentialSampler(test_tensor_data)
        test_dataloader = DataLoader(
            test_tensor_data, sampler=test_sampler, batch_size=args.batch_size, num_workers=8, pin_memory=True,
        )

        num_train_optimization_steps = (
            int(
                len(train_tensor_data)
                / args.batch_size
                / args.gradient_accumulation_steps
            )
            * args.epochs
        )
        num_warmup_steps = int(
            num_train_optimization_steps * args.warmup_proportion
        )

        linear_params = list(map(id, model.module.scoreLinear.parameters()))
        classifiar_params = list(map(id, model.module.classifier_ques.parameters()))
        base_params = filter(lambda p: id(p) not in (linear_params + classifiar_params), model.module.parameters())
        params = [{'params': base_params, 'lr': args.learning_rate},
                  {'params': model.module.classifier_ques.parameters(), 'lr': args.learning_rate*8},
                  {'params': model.module.scoreLinear.parameters(), 'lr': 100*args.learning_rate}]

        optimizer = AdamW(
            params,
            lr=args.learning_rate,
            correct_bias=False,
        )

        scheduler = WarmupLinearSchedule(
            optimizer,
            warmup_steps=num_warmup_steps,
            t_total=num_train_optimization_steps,
        )

        logger.info("  Num optimization steps = %d",
                    num_train_optimization_steps)
        logger.info("  Num warmup steps = %d", num_warmup_steps)

        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_set))
        logger.info("  Batch size = %d", args.batch_size)
        logger.info("  Gradient accumulation steps = %d",
                    args.gradient_accumulation_steps)

        best_vali_epoch_idx = -1
        best_vali_score = -1

        best_test_epoch_idx = -1
        best_test_score = -1

        for epoch_idx in range(args.epochs):
            train_loss, train_acc = train(
                model, train_dataloader, optimizer, scheduler, epoch_idx, logger, args, LEN_TRAIN_SET)
            logger.info('Epoch [{:d}/{:d}], AVG_loss: {:.9f}, Train_ACC: {:.9f}'
                        .format(epoch_idx + 1, args.epochs, train_loss, train_acc))
            vali_acc = eval(
                model, vali_dataloader, args, LEN_EVAL_SET)
            logger.info('Epoch [{:d}/{:d}], Vali_ACC: {:.9f}'
                        .format(epoch_idx + 1, args.epochs, vali_acc))
            if vali_acc >= best_vali_score:
                best_vali_score = vali_acc
                best_vali_epoch_idx = epoch_idx + 1
                torch.save(model.module.state_dict(),
                           os.path.join(PATH, "best_acc_model.pth"))

            logger.info('Epoch [{:d}/{:d}], now_best_eval_acc:{:.9f}, best_epoch:{:d}\n,'
                        .format(epoch_idx + 1, args.epochs, best_vali_score, best_vali_epoch_idx))
            params = list(model.module.scoreLinear.named_parameters())
            logger.info(params[0])

            test_acc = eval(model, test_dataloader, args, LEN_TEST_SET)
            logger.info('Epoch [{:d}/{:d}], Test_ACC: {:.9f}'
                        .format(epoch_idx + 1, args.epochs, test_acc))
            if test_acc >= best_test_score:
                best_test_score = test_acc
                best_test_epoch_idx = epoch_idx + 1
            logger.info('Epoch [{:d}/{:d}], now_best_test_acc:{:.9f}, best_epoch:{:d}\n,'
                        .format(epoch_idx + 1, args.epochs, best_test_score, best_test_epoch_idx))

    else:
        logger.info("Loading test data...")
        test_data, test_tensor_data = model.module._process_mentions_for_model(
            test_set, args.top_k, xlnet_tokenizer, longformer_tokenizer, args.max_seq_length, args.max_desc_length, args.max_q_length, args.debug, blink=args.useBLINKDic, logger=logger,
            num_of_cqa=args.max_cqa_nums, max_user_q_nums=args.max_user_q_nums, max_topic_q_nums=args.max_topic_q_nums, max_topic_nums=args.max_topic_nums,
        )
        test_sampler = SequentialSampler(test_tensor_data)
        test_dataloader = DataLoader(
            test_tensor_data, sampler=test_sampler, batch_size=args.batch_size, num_workers=8, pin_memory=True,
        )

        logger.info("*****Running Testing*****")
        logger.info("  Num examples = %d", len(test_set))
        model_state_dict = torch.load(os.path.join(PATH, "best_acc_model.pth"))
        model.module.load_state_dict(model_state_dict, False)
        test_acc = eval(model, test_dataloader, args)
        logger.info("the accuarcy of split_index {:d} is {:.9f}".format(
            args.split_index, test_acc))


if __name__ == '__main__':
    t0 = datetime.datetime.now()
    print('='*40)
    print('startTime is {:}'.format(t0))
    for i in range(1):
        parser = argparse.ArgumentParser(
            description='CQA Entity Linking Leveraging Auxiliary Data')
        parser.add_argument("--data_file", type=str,
                            default="../Data/CQAEL_update_dataset_complete.json")
        parser.add_argument("--use_topic", type=bool, default=True)
        parser.add_argument("--use_user", type=bool, default=True)
        parser.add_argument("--use_mixed_qa", type=bool, default=True)
        parser.add_argument("--use_cqa", type=bool, default=True)
        parser.add_argument("--vali", type=bool, default=True)
        parser.add_argument("--split_index", type=int, default=0)
        parser.add_argument("--cuda", type=bool, default=True)
        parser.add_argument("--top_k", type=int, default=20)
        parser.add_argument("--debug", type=bool, default=False)
        parser.add_argument("--max_seq_length", type=int, default=128)
        parser.add_argument("--max_desc_length", type=int, default=128)
        parser.add_argument("--max_q_length", type=int, default=64)
        parser.add_argument("--max_cqa_nums", type=int, default=3)
        parser.add_argument("--max_topic_q_nums", type=int, default=3)
        parser.add_argument("--max_user_q_nums", type=int, default=3)
        parser.add_argument("--max_topic_nums", type=int, default=3)
        parser.add_argument("--batch_size", type=int, default=2)
        parser.add_argument("--warmup_proportion", type=float, default=0.1)
        parser.add_argument("--gradient_accumulation_steps",
                            type=int, default=4)
        parser.add_argument(
            "--print_tr_loss_opt_steps_interval", type=int, default=2)
        parser.add_argument("--max_grad_norm", type=float, default=1.0)
        parser.add_argument("--learning_rate", type=float, default=1e-5)
        parser.add_argument("--epochs", type=int, default=10)
        parser.add_argument("--out_dim", type=int, default=768)
        parser.add_argument("--add_linear", type=bool, default=False)
        parser.add_argument("--store_file", type=str, default="Logger")
        parser.add_argument("--model_storage_path",
                            type=str, default="../best_model")
        parser.add_argument('--local_rank', default=-1, type=int,
                            help='node rank for distributed training')
        parser.add_argument("--seed", type=int, default=12345)
        parser.add_argument("--train_and_eval", type=bool, default=True)
        parser.add_argument("--useBLINKDic", type=bool, default=True)

        parser.add_argument("--max_posts", type=int, default=20)
        parser.add_argument("--max_len", type=int, default=32)
        parser.add_argument("--XP", type=bool, default=True)
        parser.add_argument("--start_step", type=int, default=9)
        args = parser.parse_args()
        main(args)

