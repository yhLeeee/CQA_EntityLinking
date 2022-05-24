import torch.nn.functional as F
from torch.utils.data import TensorDataset
from tqdm import tqdm
import torch.nn as nn
import torch
from transformers import LongformerConfig, LongformerModel, LongformerTokenizer, XLNetModel, XLNetConfig, XLNetTokenizer
import copy
import heapq
import Levenshtein
import difflib

START_DESC = '<d>'
END_DESC = '</d>'
START_QUES = '<q>'
END_QUES = '</q>'
SEP_TOKEN = '<sep>'
ENT_TOKEN = '<ent>'

EXTRA_TOKENS = [
    START_DESC,
    END_DESC,
    START_QUES,
    END_QUES,
    SEP_TOKEN,
    ENT_TOKEN
]

class CQAEL(nn.Module):
    def __init__(self, args):
        super(CQAEL, self).__init__()
        self.top_k = args.top_k
        self.use_QA = args.use_mixed_qa
        self.use_topic = args.use_topic
        self.use_user = args.use_user
        # self-attention on QA_sentences
        self.use_cqa = args.use_cqa

        # tokenizer
        self.xlnet_tokenizer = XLNetTokenizer.from_pretrained(
            '../Data/xlnet-base-cased')

        # use to classification <ctxt, desc>
        self.XLNet_config = XLNetConfig.from_pretrained(
            '../Data/xlnet-base-cased/config.json')
        self.XLNet_config.hyper = args
        # two cross-encoders
        self.c_encoder = XLNetModel.from_pretrained(
            '../Data/xlnet-base-cased/pytorch_model.bin', config=self.XLNet_config)
        
        # tokenizer
        self.longformer_tokenizer = LongformerTokenizer.from_pretrained('../Data/longformer-base-4096', additional_special_tokens=EXTRA_TOKENS)
        
        # q_encoder
        longformer_config_q = LongformerConfig.from_pretrained(
            '../Data/longformer-base-4096/config.json',
            attention_mode='sliding_chunks_no_overlap',
            attention_dropout=0.1)
        attention_size = 16
        longformer_config_q.attention_window = [attention_size] * longformer_config_q.num_hidden_layers

        self.q_encoder = LongformerModel.from_pretrained(
            '../Data/longformer-base-4096/pytorch_model.bin', config=longformer_config_q)
        self.q_encoder.resize_token_embeddings(len(self.longformer_tokenizer.get_vocab()))

        self.max_topic_nums = args.max_topic_nums
        self.max_topic_q_nums = args.max_topic_q_nums
        self.max_user_q_nums = args.max_user_q_nums
        self.max_seq_length = args.max_seq_length
        self.max_q_length = args.max_q_length
        self.max_cqa_nums = args.max_cqa_nums

        self.classifier_ctxt = nn.Linear(768, 1)
        self.classifier_ques = nn.Linear(768, 1)

        if args.use_topic and args.use_user and args.use_cqa:
            self.scoreLinear = nn.Linear(5, 1, bias=False)
        elif args.use_topic == False and args.use_user == False and args.use_cqa == False:
            self.scoreLinear = nn.Linear(2, 1, bias=False)
        elif (args.use_topic == False and args.use_user == False and args.use_cqa == True) or \
                (args.use_topic == True and args.use_user == False and args.use_cqa == False) or \
                (args.use_topic == False and args.use_user == True and args.use_cqa == False):
            self.scoreLinear = nn.Linear(3, 1, bias=False)
        else:
            self.scoreLinear = nn.Linear(4, 1, bias=False)

        nn.init.constant_(self.scoreLinear.weight, 1)

        self.dropout = nn.Dropout(0.1)
        self.lossFun = nn.CrossEntropyLoss()

    def forward(
            self,
            all_ctxt_cross_ids=None,
            all_ctxt_cross_mask=None,
            all_ctxt_cross_seg=None,
            all_cqa_cross_ids=None,
            all_topic_cross_ids=None,
            all_user_cross_ids=None,
            all_uesr_flag=None,
            candidate_priors=None,
            labels=None,
            entity_mask=None,
    ):
        # 2 features
        if not self.use_QA and not self.use_user and not self.use_topic and not self.use_cqa:
            self.c_encoder.hyper_config.XP = False
            bsz, _, qa_max_len = all_ctxt_cross_ids.shape
            flat_input_ctxt_ids = all_ctxt_cross_ids.view(-1, qa_max_len)
            flat_input_ctxt_mask = all_ctxt_cross_mask.view(-1, qa_max_len)
            flat_input_ctxt_seg = all_ctxt_cross_seg.view(-1, qa_max_len)

            ctxt_output = self.c_encoder(
                input_ids=flat_input_ctxt_ids,
                token_type_ids=flat_input_ctxt_seg,
                attention_mask=flat_input_ctxt_mask,
            )

            ctxt_embs = ctxt_output[0][:, 0].view(bsz, -1, 768)

            ctxt_scores = self.classifier_ctxt(ctxt_embs)
            ctxt_scores = torch.squeeze(ctxt_scores, dim=2)

            final_score_vec = torch.stack(
                (ctxt_scores, candidate_priors), dim=2)
            final_score = self.scoreLinear(final_score_vec)
            reshaped_logits = torch.squeeze(final_score, dim=2)

            entity_mask = (1.0 - entity_mask) * -1000
            reshaped_logits = reshaped_logits + entity_mask

            outputs = (reshaped_logits,)
            loss = self.lossFun(reshaped_logits, labels)
            outputs = (loss,) + outputs

            return outputs

        # 3 features (base + topic)
        if not self.use_QA and not self.use_user and self.use_topic and not self.use_cqa:
            self.c_encoder.hyper_config.XP = False
            bsz, _, qa_max_len = all_ctxt_cross_ids.shape
            flat_input_ctxt_ids = all_ctxt_cross_ids.view(-1, qa_max_len)
            flat_input_ctxt_mask = all_ctxt_cross_mask.view(-1, qa_max_len)
            flat_input_ctxt_seg = all_ctxt_cross_seg.view(-1, qa_max_len)

            ctxt_output = self.c_encoder(
                input_ids=flat_input_ctxt_ids,
                token_type_ids=flat_input_ctxt_seg,
                attention_mask=flat_input_ctxt_mask,
            )

            ctxt_embs = ctxt_output[0][:, 0].view(bsz, -1, 768)

            ctxt_scores = self.classifier_ctxt(ctxt_embs)
            ctxt_scores = torch.squeeze(ctxt_scores, dim=2)


            flat_all_topic_cross_ids = all_topic_cross_ids.view(-1, all_topic_cross_ids.size(-1))
            attention_mask = torch.ones(flat_all_topic_cross_ids.shape, dtype=torch.long)
            attention_mask[flat_all_topic_cross_ids == self.longformer_tokenizer.pad_token_id] = 0
            attention_mask[0, :self.max_seq_length] = 2
            attention_mask = attention_mask.cuda()
            global_attention_mask = [1] * 1 + [0] * (flat_all_topic_cross_ids.size(-1) - 1)
            global_attention_mask = torch.tensor(global_attention_mask, dtype=torch.long).unsqueeze(0)
            global_attention_mask = global_attention_mask.cuda()

            outputs = self.q_encoder(
                flat_all_topic_cross_ids, 
                attention_mask=attention_mask,
                global_attention_mask=global_attention_mask,
            )
            topic_output = outputs.last_hidden_state[:, 0, :].view(bsz, -1, 768)
            topic_score = self.classifier_ques(topic_output)

            final_score_vec = torch.stack(
                (ctxt_scores, candidate_priors), dim=2)
            final_score_vec = torch.cat(
                (final_score_vec, topic_score), 2)
            final_score = self.scoreLinear(final_score_vec)
            reshaped_logits = torch.squeeze(final_score, dim=2)

            entity_mask = (1.0 - entity_mask) * -1000
            reshaped_logits = reshaped_logits + entity_mask

            outputs = (reshaped_logits,)
            loss = self.lossFun(reshaped_logits, labels)
            outputs = (loss,) + outputs

            return outputs
        
        # 3 features (base + user)
        if not self.use_QA and self.use_user and not self.use_topic and not self.use_cqa:
            self.c_encoder.hyper_config.XP = False
            bsz, _, qa_max_len = all_ctxt_cross_ids.shape
            flat_input_ctxt_ids = all_ctxt_cross_ids.view(-1, qa_max_len)
            flat_input_ctxt_mask = all_ctxt_cross_mask.view(-1, qa_max_len)
            flat_input_ctxt_seg = all_ctxt_cross_seg.view(-1, qa_max_len)

            ctxt_output = self.c_encoder(
                input_ids=flat_input_ctxt_ids,
                token_type_ids=flat_input_ctxt_seg,
                attention_mask=flat_input_ctxt_mask,
            )

            ctxt_embs = ctxt_output[0][:, 0].view(bsz, -1, 768)

            ctxt_scores = self.classifier_ctxt(ctxt_embs)
            ctxt_scores = torch.squeeze(ctxt_scores, dim=2)


            # uesr_scores
            flat_all_user_cross_ids = all_user_cross_ids.view(-1, all_user_cross_ids.size(-1))
            attention_mask = torch.ones(flat_all_user_cross_ids.shape, dtype=torch.long)
            attention_mask[flat_all_user_cross_ids == self.longformer_tokenizer.pad_token_id] = 0
            attention_mask[0, :(self.max_seq_length//2)] = 2
            attention_mask = attention_mask.cuda()
            global_attention_mask = [1] * 1 + [0] * (flat_all_user_cross_ids.size(-1) - 1)
            global_attention_mask = torch.tensor(global_attention_mask, dtype=torch.long).unsqueeze(0)
            global_attention_mask = global_attention_mask.cuda()

            outputs = self.q_encoder(
                flat_all_user_cross_ids, 
                attention_mask=attention_mask,
                global_attention_mask=global_attention_mask,
            )
            user_output = outputs.last_hidden_state[:, 0, :].view(bsz, -1, 768)
            user_score = self.classifier_ques(user_output)
            user_score = torch.squeeze(user_score, dim=2)
            user_score = user_score * all_uesr_flag
            user_score = torch.unsqueeze(user_score, dim=2)

            final_score_vec = torch.stack(
                (ctxt_scores, candidate_priors), dim=2)
            final_score_vec = torch.cat(
                (final_score_vec, user_score), 2)
            final_score = self.scoreLinear(final_score_vec)
            reshaped_logits = torch.squeeze(final_score, dim=2)

            entity_mask = (1.0 - entity_mask) * -1000
            reshaped_logits = reshaped_logits + entity_mask

            outputs = (reshaped_logits,)
            loss = self.lossFun(reshaped_logits, labels)
            outputs = (loss,) + outputs

            return outputs

        # 3 features (base + cqa)
        if not self.use_QA and not self.use_user and not self.use_topic and self.use_cqa:
            self.c_encoder.hyper_config.XP = False
            bsz, _, qa_max_len = all_ctxt_cross_ids.shape
            flat_input_ctxt_ids = all_ctxt_cross_ids.view(-1, qa_max_len)
            flat_input_ctxt_mask = all_ctxt_cross_mask.view(-1, qa_max_len)
            flat_input_ctxt_seg = all_ctxt_cross_seg.view(-1, qa_max_len)

            ctxt_output = self.c_encoder(
                input_ids=flat_input_ctxt_ids,
                token_type_ids=flat_input_ctxt_seg,
                attention_mask=flat_input_ctxt_mask,
            )

            ctxt_embs = ctxt_output[0][:, 0].view(bsz, -1, 768)

            ctxt_scores = self.classifier_ctxt(ctxt_embs)
            ctxt_scores = torch.squeeze(ctxt_scores, dim=2)

            flat_all_cqa_cross_ids = all_cqa_cross_ids.view(-1, all_cqa_cross_ids.size(-1))
            attention_mask = torch.ones(flat_all_cqa_cross_ids.shape, dtype=torch.long)
            attention_mask[flat_all_cqa_cross_ids == self.longformer_tokenizer.pad_token_id] = 0
            attention_mask[0, :(self.max_seq_length//2)] = 2
            attention_mask = attention_mask.cuda()
            global_attention_mask = [1] * 1 + [0] * (flat_all_cqa_cross_ids.size(-1) - 1)
            global_attention_mask = torch.tensor(global_attention_mask, dtype=torch.long).unsqueeze(0)
            global_attention_mask = global_attention_mask.cuda()

            outputs = self.q_encoder(
                flat_all_cqa_cross_ids, 
                attention_mask=attention_mask,
                global_attention_mask=global_attention_mask,
            )
            # bsz * 20 * 768
            cqa_output = outputs.last_hidden_state[:, 0, :].view(bsz, -1, 768)
            # bsz * 20 * 1
            cqa_scores = self.classifier_ques(cqa_output)

            final_score_vec = torch.stack(
                (ctxt_scores, candidate_priors), dim=2)
            final_score_vec = torch.cat(
                (final_score_vec, cqa_scores), 2)
            final_score = self.scoreLinear(final_score_vec)
            reshaped_logits = torch.squeeze(final_score, dim=2)

            entity_mask = (1.0 - entity_mask) * -1000
            reshaped_logits = reshaped_logits + entity_mask

            outputs = (reshaped_logits,)
            loss = self.lossFun(reshaped_logits, labels)
            outputs = (loss,) + outputs

            return outputs

        # 4 features (base + cqa + topic)
        if not self.use_QA and not self.use_user and self.use_topic and self.use_cqa:
            # ctxt_scores
            self.c_encoder.hyper_config.XP = False
            bsz, _, qa_max_len = all_ctxt_cross_ids.shape
            flat_input_ctxt_ids = all_ctxt_cross_ids.view(-1, qa_max_len)
            flat_input_ctxt_mask = all_ctxt_cross_mask.view(-1, qa_max_len)
            flat_input_ctxt_seg = all_ctxt_cross_seg.view(-1, qa_max_len)

            ctxt_output = self.c_encoder(
                input_ids=flat_input_ctxt_ids,
                token_type_ids=flat_input_ctxt_seg,
                attention_mask=flat_input_ctxt_mask,
            )

            ctxt_embs = ctxt_output[0][:, 0].view(bsz, -1, 768)

            ctxt_scores = self.classifier_ctxt(ctxt_embs)
            ctxt_scores = torch.squeeze(ctxt_scores, dim=2)

            flat_all_cqa_cross_ids = all_cqa_cross_ids.view(-1, all_cqa_cross_ids.size(-1))
            attention_mask = torch.ones(flat_all_cqa_cross_ids.shape, dtype=torch.long)
            attention_mask[flat_all_cqa_cross_ids == self.longformer_tokenizer.pad_token_id] = 0
            attention_mask[0, :(self.max_seq_length//2)] = 2
            attention_mask = attention_mask.cuda()
            global_attention_mask = [1] * 1 + [0] * (flat_all_cqa_cross_ids.size(-1) - 1)
            global_attention_mask = torch.tensor(global_attention_mask, dtype=torch.long).unsqueeze(0)
            global_attention_mask = global_attention_mask.cuda()

            outputs = self.q_encoder(
                flat_all_cqa_cross_ids, 
                attention_mask=attention_mask,
                global_attention_mask=global_attention_mask,
            )
            cqa_output = outputs.last_hidden_state[:, 0, :].view(bsz, -1, 768)
            cqa_scores = self.classifier_ques(cqa_output)

            flat_all_topic_cross_ids = all_topic_cross_ids.view(-1, all_topic_cross_ids.size(-1))
            attention_mask = torch.ones(flat_all_topic_cross_ids.shape, dtype=torch.long)
            attention_mask[flat_all_topic_cross_ids == self.longformer_tokenizer.pad_token_id] = 0
            attention_mask[0, :(self.max_seq_length//2)] = 2
            attention_mask = attention_mask.cuda()
            global_attention_mask = [1] * 1 + [0] * (flat_all_topic_cross_ids.size(-1) - 1)
            global_attention_mask = torch.tensor(global_attention_mask, dtype=torch.long).unsqueeze(0)
            global_attention_mask = global_attention_mask.cuda()

            outputs = self.q_encoder(
                flat_all_topic_cross_ids, 
                attention_mask=attention_mask,
                global_attention_mask=global_attention_mask,
            )
            topic_output = outputs.last_hidden_state[:, 0, :].view(bsz, -1, 768)
            topic_score = self.classifier_ques(topic_output)

            final_score_vec = torch.stack(
                (ctxt_scores, candidate_priors), dim=2)
            final_score_vec = torch.cat(
                (final_score_vec, cqa_scores), 2)
            final_score_vec = torch.cat(
                (final_score_vec, topic_score), 2)
            final_score = self.scoreLinear(final_score_vec)
            reshaped_logits = torch.squeeze(final_score, dim=2)

            entity_mask = (1.0 - entity_mask) * -1000
            reshaped_logits = reshaped_logits + entity_mask

            outputs = (reshaped_logits,)
            loss = self.lossFun(reshaped_logits, labels)
            outputs = (loss,) + outputs

            return outputs

        # 4 features (base + cqa + user)
        if not self.use_QA and self.use_user and not self.use_topic and self.use_cqa:
            self.c_encoder.hyper_config.XP = False
            bsz, _, qa_max_len = all_ctxt_cross_ids.shape
            flat_input_ctxt_ids = all_ctxt_cross_ids.view(-1, qa_max_len)
            flat_input_ctxt_mask = all_ctxt_cross_mask.view(-1, qa_max_len)
            flat_input_ctxt_seg = all_ctxt_cross_seg.view(-1, qa_max_len)

            ctxt_output = self.c_encoder(
                input_ids=flat_input_ctxt_ids,
                token_type_ids=flat_input_ctxt_seg,
                attention_mask=flat_input_ctxt_mask,
            )

            ctxt_embs = ctxt_output[0][:, 0].view(bsz, -1, 768)

            ctxt_scores = self.classifier_ctxt(ctxt_embs)
            ctxt_scores = torch.squeeze(ctxt_scores, dim=2)

            flat_all_cqa_cross_ids = all_cqa_cross_ids.view(-1, all_cqa_cross_ids.size(-1))
            attention_mask = torch.ones(flat_all_cqa_cross_ids.shape, dtype=torch.long)
            attention_mask[flat_all_cqa_cross_ids == self.longformer_tokenizer.pad_token_id] = 0
            attention_mask[0, :(self.max_seq_length//2)] = 2
            attention_mask = attention_mask.cuda()
            global_attention_mask = [1] * 1 + [0] * (flat_all_cqa_cross_ids.size(-1) - 1)
            global_attention_mask = torch.tensor(global_attention_mask, dtype=torch.long).unsqueeze(0)
            global_attention_mask = global_attention_mask.cuda()

            outputs = self.q_encoder(
                flat_all_cqa_cross_ids, 
                attention_mask=attention_mask,
                global_attention_mask=global_attention_mask,
            )
            cqa_output = outputs.last_hidden_state[:, 0, :].view(bsz, -1, 768)
            cqa_scores = self.classifier_ques(cqa_output)

            flat_all_user_cross_ids = all_user_cross_ids.view(-1, all_user_cross_ids.size(-1))
            attention_mask = torch.ones(flat_all_user_cross_ids.shape, dtype=torch.long)
            attention_mask[flat_all_user_cross_ids == self.longformer_tokenizer.pad_token_id] = 0
            attention_mask[0, :self.max_seq_length] = 2
            attention_mask = attention_mask.cuda()
            global_attention_mask = [1] * 1 + [0] * (flat_all_user_cross_ids.size(-1) - 1)
            global_attention_mask = torch.tensor(global_attention_mask, dtype=torch.long).unsqueeze(0)
            global_attention_mask = global_attention_mask.cuda()

            outputs = self.q_encoder(
                flat_all_user_cross_ids, 
                attention_mask=attention_mask,
                global_attention_mask=global_attention_mask,
            )
            user_output = outputs.last_hidden_state[:, 0, :].view(bsz, -1, 768)
            user_score = self.classifier_ques(user_output)
            user_score = torch.squeeze(user_score, dim=2)
            user_score = user_score * all_uesr_flag
            user_score = torch.unsqueeze(user_score, dim=2)

            final_score_vec = torch.stack(
                (ctxt_scores, candidate_priors), dim=2)
            final_score_vec = torch.cat(
                (final_score_vec, cqa_scores), 2)
            final_score_vec = torch.cat(
                (final_score_vec, user_score), 2)
            final_score = self.scoreLinear(final_score_vec)
            reshaped_logits = torch.squeeze(final_score, dim=2)

            entity_mask = (1.0 - entity_mask) * -1000
            reshaped_logits = reshaped_logits + entity_mask

            outputs = (reshaped_logits,)
            loss = self.lossFun(reshaped_logits, labels)
            outputs = (loss,) + outputs

            return outputs


        # 4 features (base + topic + user)
        if not self.use_QA and self.use_user and self.use_topic and not self.use_cqa:
            self.c_encoder.hyper_config.XP = False
            bsz, _, qa_max_len = all_ctxt_cross_ids.shape
            flat_input_ctxt_ids = all_ctxt_cross_ids.view(-1, qa_max_len)
            flat_input_ctxt_mask = all_ctxt_cross_mask.view(-1, qa_max_len)
            flat_input_ctxt_seg = all_ctxt_cross_seg.view(-1, qa_max_len)

            ctxt_output = self.c_encoder(
                input_ids=flat_input_ctxt_ids,
                token_type_ids=flat_input_ctxt_seg,
                attention_mask=flat_input_ctxt_mask,
            )

            ctxt_embs = ctxt_output[0][:, 0].view(bsz, -1, 768)

            ctxt_scores = self.classifier_ctxt(ctxt_embs)
            ctxt_scores = torch.squeeze(ctxt_scores, dim=2)
            
            flat_all_topic_cross_ids = all_topic_cross_ids.view(-1, all_topic_cross_ids.size(-1))
            attention_mask = torch.ones(flat_all_topic_cross_ids.shape, dtype=torch.long)
            attention_mask[flat_all_topic_cross_ids == self.longformer_tokenizer.pad_token_id] = 0
            attention_mask[0, :self.max_seq_length] = 2
            attention_mask = attention_mask.cuda()
            global_attention_mask = [1] * 1 + [0] * (flat_all_topic_cross_ids.size(-1) - 1)
            global_attention_mask = torch.tensor(global_attention_mask, dtype=torch.long).unsqueeze(0)
            global_attention_mask = global_attention_mask.cuda()

            outputs = self.q_encoder(
                flat_all_topic_cross_ids, 
                attention_mask=attention_mask,
                global_attention_mask=global_attention_mask,
            )
            topic_output = outputs.last_hidden_state[:, 0, :].view(bsz, -1, 768)
            topic_score = self.classifier_ques(topic_output)

            flat_all_user_cross_ids = all_user_cross_ids.view(-1, all_user_cross_ids.size(-1))
            attention_mask = torch.ones(flat_all_user_cross_ids.shape, dtype=torch.long)
            attention_mask[flat_all_user_cross_ids == self.longformer_tokenizer.pad_token_id] = 0
            attention_mask[0, :self.max_seq_length] = 2
            attention_mask = attention_mask.cuda()
            global_attention_mask = [1] * 1 + [0] * (flat_all_user_cross_ids.size(-1) - 1)
            global_attention_mask = torch.tensor(global_attention_mask, dtype=torch.long).unsqueeze(0)
            global_attention_mask = global_attention_mask.cuda()

            outputs = self.q_encoder(
                flat_all_user_cross_ids, 
                attention_mask=attention_mask,
                global_attention_mask=global_attention_mask,
            )
            user_output = outputs.last_hidden_state[:, 0, :].view(bsz, -1, 768)
            user_score = self.classifier_ques(user_output)
            user_score = torch.squeeze(user_score, dim=2)
            user_score = user_score * all_uesr_flag
            user_score = torch.unsqueeze(user_score, dim=2)

            final_score_vec = torch.stack(
                (ctxt_scores, candidate_priors), dim=2)
            final_score_vec = torch.cat(
                (final_score_vec, topic_score), 2)
            final_score_vec = torch.cat(
                (final_score_vec, user_score), 2)
            final_score = self.scoreLinear(final_score_vec)
            reshaped_logits = torch.squeeze(final_score, dim=2)

            entity_mask = (1.0 - entity_mask) * -1000
            reshaped_logits = reshaped_logits + entity_mask

            outputs = (reshaped_logits,)
            loss = self.lossFun(reshaped_logits, labels)
            outputs = (loss,) + outputs

            return outputs

        # 5 features (base + cqa + topic + user)
        if not self.use_QA and self.use_user and self.use_topic and self.use_cqa:
            self.c_encoder.hyper_config.XP = False
            bsz, _, qa_max_len = all_ctxt_cross_ids.shape
            flat_input_ctxt_ids = all_ctxt_cross_ids.view(-1, qa_max_len)
            flat_input_ctxt_mask = all_ctxt_cross_mask.view(-1, qa_max_len)
            flat_input_ctxt_seg = all_ctxt_cross_seg.view(-1, qa_max_len)

            ctxt_output = self.c_encoder(
                input_ids=flat_input_ctxt_ids,
                token_type_ids=flat_input_ctxt_seg,
                attention_mask=flat_input_ctxt_mask,
            )

            ctxt_embs = ctxt_output[0][:, 0].view(bsz, -1, 768)

            ctxt_scores = self.classifier_ctxt(ctxt_embs)
            ctxt_scores = torch.squeeze(ctxt_scores, dim=2)

            flat_all_cqa_cross_ids = all_cqa_cross_ids.view(-1, all_cqa_cross_ids.size(-1))
            attention_mask = torch.ones(flat_all_cqa_cross_ids.shape, dtype=torch.long)
            attention_mask[flat_all_cqa_cross_ids == self.longformer_tokenizer.pad_token_id] = 0
            attention_mask[0, :(self.max_seq_length//2)] = 2
            attention_mask = attention_mask.cuda()
            global_attention_mask = [1] * 1 + [0] * (flat_all_cqa_cross_ids.size(-1) - 1)
            global_attention_mask = torch.tensor(global_attention_mask, dtype=torch.long).unsqueeze(0)
            global_attention_mask = global_attention_mask.cuda()

            outputs = self.q_encoder(
                flat_all_cqa_cross_ids, 
                attention_mask=attention_mask,
                global_attention_mask=global_attention_mask,
            )
            cqa_output = outputs.last_hidden_state[:, 0, :].view(bsz, -1, 768)
            cqa_scores = self.classifier_ques(cqa_output)

            flat_all_topic_cross_ids = all_topic_cross_ids.view(-1, all_topic_cross_ids.size(-1))
            attention_mask = torch.ones(flat_all_topic_cross_ids.shape, dtype=torch.long)
            attention_mask[flat_all_topic_cross_ids == self.longformer_tokenizer.pad_token_id] = 0
            attention_mask[0, :(self.max_seq_length//2)] = 2
            attention_mask = attention_mask.cuda()
            global_attention_mask = [1] * 1 + [0] * (flat_all_topic_cross_ids.size(-1) - 1)
            global_attention_mask = torch.tensor(global_attention_mask, dtype=torch.long).unsqueeze(0)
            global_attention_mask = global_attention_mask.cuda()

            outputs = self.q_encoder(
                flat_all_topic_cross_ids, 
                attention_mask=attention_mask,
                global_attention_mask=global_attention_mask,
            )
            topic_output = outputs.last_hidden_state[:, 0, :].view(bsz, -1, 768)
            topic_score = self.classifier_ques(topic_output)


            flat_all_user_cross_ids = all_user_cross_ids.view(-1, all_user_cross_ids.size(-1))
            attention_mask = torch.ones(flat_all_user_cross_ids.shape, dtype=torch.long)
            attention_mask[flat_all_user_cross_ids == self.longformer_tokenizer.pad_token_id] = 0
            attention_mask[0, :self.max_seq_length] = 2
            attention_mask = attention_mask.cuda()
            global_attention_mask = [1] * 1 + [0] * (flat_all_user_cross_ids.size(-1) - 1)
            global_attention_mask = torch.tensor(global_attention_mask, dtype=torch.long).unsqueeze(0)
            global_attention_mask = global_attention_mask.cuda()

            outputs = self.q_encoder(
                flat_all_user_cross_ids, 
                attention_mask=attention_mask,
                global_attention_mask=global_attention_mask,
            )
            user_output = outputs.last_hidden_state[:, 0, :].view(bsz, -1, 768)
            user_score = self.classifier_ques(user_output)
            user_score = torch.squeeze(user_score, dim=2)
            user_score = user_score * all_uesr_flag
            user_score = torch.unsqueeze(user_score, dim=2)

            final_score_vec = torch.stack(
                (ctxt_scores, candidate_priors), dim=2)
            final_score_vec = torch.cat(
                (final_score_vec, cqa_scores), 2)
            final_score_vec = torch.cat(
                (final_score_vec, topic_score), 2)
            final_score_vec = torch.cat(
                (final_score_vec, user_score), 2)
            final_score = self.scoreLinear(final_score_vec)
            reshaped_logits = torch.squeeze(final_score, dim=2)

            entity_mask = (1.0 - entity_mask) * -1000
            reshaped_logits = reshaped_logits + entity_mask

            outputs = (reshaped_logits,)
            loss = self.lossFun(reshaped_logits, labels)
            outputs = (loss,) + outputs

            return outputs

    @staticmethod
    def _select_field(samples, field):
        return [
            [cand[field] for cand in sample["candidate_features"]] for sample in samples
        ]

    @staticmethod
    def _get_candidate_tokens_representation(
            candidate_prior,
            xlnet_tokenizer,
            longformer_tokenizer,
            candidate_title,
            candidate_desc,
            sample,
            max_seq_length,
            max_desc_length,
            max_q_length,
            num_of_cqa,
            num_of_user_q,
            num_of_topic_q,
            num_of_topic,
    ):
        max_sub_seq_length = (max_seq_length - 3) // 2
        candidate_title_tokens = xlnet_tokenizer.tokenize(candidate_title)
        candidate_desc_tokens = xlnet_tokenizer.tokenize(candidate_desc)
        max_desc_seq_length = max_sub_seq_length - len(candidate_title_tokens)
        if len(candidate_desc_tokens) > max_desc_seq_length:
            candidate_desc_tokens = candidate_desc_tokens[:max_desc_seq_length]
        cand_tokens = (candidate_title_tokens + ["[ENT]"] + candidate_desc_tokens)
        context = sample["qa_sent"]
        context_tokens = xlnet_tokenizer.tokenize(context)
        if len(context_tokens) > max_sub_seq_length:
            context_tokens = context_tokens[:max_sub_seq_length]

        tokens = (["[CLS]"] + context_tokens + ["[SEP]"] +
                  cand_tokens + ["[SEP]"])
        ctxt_cross_seg = [0] * (len(context_tokens) + 2) + \
            [1] * (len(cand_tokens) + 1)
        ctxt_cross_ids = xlnet_tokenizer.convert_tokens_to_ids(tokens)
        ctxt_cross_mask = [1] * len(ctxt_cross_ids)

        padding = [0] * (max_seq_length - len(ctxt_cross_ids))
        ctxt_cross_ids += padding
        ctxt_cross_mask += padding
        ctxt_cross_seg += padding

        assert len(ctxt_cross_ids) == max_seq_length
        assert len(ctxt_cross_mask) == max_seq_length
        assert len(ctxt_cross_seg) == max_seq_length

        bos = longformer_tokenizer.bos_token_id
        eos = longformer_tokenizer.eos_token_id

        candidate_title_desc = START_DESC + candidate_title + ENT_TOKEN + candidate_desc + END_DESC
        candidate_tokens = longformer_tokenizer.encode(
            candidate_title_desc,
            add_special_tokens=False,
            truncation=max_desc_length-2 is not None,
            max_length=max_desc_length-2,
            padding='max_length'
        )
        candidate_tokens = [bos] + candidate_tokens
        if candidate_tokens[-1] == longformer_tokenizer.encode(END_DESC, add_special_tokens=False)[0]:
            candidate_tokens += [1]
        else:
            candidate_tokens += [longformer_tokenizer.encode(END_DESC, add_special_tokens=False)[0]]

        all_cqa = []
        for cqa_sent in sample["cqa_sentences"]:
            all_cqa.append(cqa_sent)

        all_cqa = list(set(all_cqa))
        if len(all_cqa) > 1:
            for cqa in all_cqa:
                if cqa == context:
                    all_cqa.remove(cqa)
        for cqa in all_cqa:
            cqa = START_QUES + cqa + END_QUES
        
        string_simi_scores = []
        for cqa in all_cqa:
            score1 = Levenshtein.ratio(context, cqa)
            score2 = Levenshtein.jaro_winkler(context, cqa)
            score3 = difflib.SequenceMatcher(context, cqa).ratio()
            score_avg = (score1 + score2 + score3) / 3
            string_simi_scores.append(score_avg)
        index = heapq.nlargest(num_of_cqa, range(len(string_simi_scores)), string_simi_scores.__getitem__)

        all_cqa_tokens = []
        for i in index:
            cqa_tokens = longformer_tokenizer.encode(
                    all_cqa[i],
                    add_special_tokens=False,
                    truncation=max_q_length-1 is not None,
                    max_length=max_q_length-1 ,
                    padding='max_length'
                )
            if cqa_tokens[-1] == longformer_tokenizer.encode(END_QUES, add_special_tokens=False)[0]:
                    cqa_tokens += [1]
            else:
                cqa_tokens += [longformer_tokenizer.encode(END_QUES, add_special_tokens=False)[0]] 
            assert len(cqa_tokens) == max_q_length
            all_cqa_tokens.append(cqa_tokens)
        
        padding = [1] * max_q_length
        if len(all_cqa_tokens) < num_of_cqa:
            for _ in range(num_of_cqa - len(all_cqa_tokens)):
                all_cqa_tokens.append(padding)
        assert len(all_cqa_tokens) == num_of_cqa

        cqa_cross_ids = copy.deepcopy(candidate_tokens)
        cqa_cross_ids += [eos]
        for i in range(len(all_cqa_tokens)):
            cqa_cross_ids += all_cqa_tokens[i]
        cqa_cross_ids += [eos]

        all_topic_ques = []
        for topic in sample["topic_meta_data"]:
            topic_name = topic["topic_name"]
            for q in topic["topic_questions"]:
                ques = START_QUES + topic_name + SEP_TOKEN + q + END_QUES
                all_topic_ques.append(ques)
        
        context = sample["qa_sent"]
        string_simi_scores = []
        for ques in all_topic_ques:
            score1 = Levenshtein.ratio(context, ques)
            score2 = Levenshtein.jaro_winkler(context, ques)
            score3 = difflib.SequenceMatcher(context, ques).ratio()
            score_avg = (score1 + score2 + score3) / 3
            string_simi_scores.append(score_avg)
        index = heapq.nlargest(num_of_topic_q, range(len(string_simi_scores)), string_simi_scores.__getitem__)

        all_topic_tokens = []
        for i in index:
            ques_tokens = longformer_tokenizer.encode(
                    all_topic_ques[i],
                    add_special_tokens=False,
                    truncation=max_q_length-1 is not None,
                    max_length=max_q_length-1 ,
                    padding='max_length'
                )
            if ques_tokens[-1] == longformer_tokenizer.encode(END_QUES, add_special_tokens=False)[0]:
                    ques_tokens += [1]
            else:
                ques_tokens += [longformer_tokenizer.encode(END_QUES, add_special_tokens=False)[0]] 
            assert len(ques_tokens) == max_q_length
            all_topic_tokens.append(ques_tokens)
        
        padding = [1] * max_q_length
        if len(all_topic_tokens) < num_of_topic_q:
            for _ in range(num_of_topic_q - len(all_topic_tokens)):
                all_topic_tokens.append(padding)
        assert len(all_topic_tokens) == num_of_topic_q

        topic_cross_ids = copy.deepcopy(candidate_tokens)
        topic_cross_ids += [eos]
        for i in range(len(all_topic_tokens)):
            topic_cross_ids += all_topic_tokens[i]
        topic_cross_ids += [eos]

        all_user_ques = []
        for q in sample["user_meta_data"]:
            ques = START_QUES + q + END_QUES
            all_user_ques.append(ques)
        
        context = sample["qa_sent"]
        string_simi_scores = []
        for ques in all_user_ques:
            # three types of string similarity score
            score1 = Levenshtein.ratio(context, ques)
            score2 = Levenshtein.jaro_winkler(context, ques)
            score3 = difflib.SequenceMatcher(context, ques).ratio()
            score_avg = (score1 + score2 + score3) / 3
            string_simi_scores.append(score_avg)
        index = heapq.nlargest(num_of_user_q, range(len(string_simi_scores)), string_simi_scores.__getitem__)

        all_user_tokens = []
        for i in index:
            ques_tokens = longformer_tokenizer.encode(
                    all_user_ques[i],
                    add_special_tokens=False,
                    truncation=max_q_length-1 is not None,
                    max_length=max_q_length-1 ,
                    padding='max_length'
                )
            if ques_tokens[-1] == longformer_tokenizer.encode(END_QUES, add_special_tokens=False)[0]:
                    ques_tokens += [1]
            else:
                ques_tokens += [longformer_tokenizer.encode(END_QUES, add_special_tokens=False)[0]] 
            assert len(ques_tokens) == max_q_length
            all_user_tokens.append(ques_tokens)

        using_user = 1
        padding = [1] * max_q_length
        if len(all_user_tokens) == 0:
            using_user = 0
            for _ in range(num_of_user_q):
                all_user_tokens.append(padding)
        if len(all_user_tokens) < num_of_user_q:
            for _ in range(num_of_user_q - len(all_user_tokens)):
                all_user_tokens.append(padding)
        assert len(all_user_tokens) == num_of_user_q

        user_cross_ids = copy.deepcopy(candidate_tokens)
        user_cross_ids += [eos]
        for i in range(len(all_user_tokens)):
            user_cross_ids += all_user_tokens[i]
        user_cross_ids += [eos]

        return {
            "ctxt_cross_ids": ctxt_cross_ids,
            "ctxt_cross_mask": ctxt_cross_mask,
            "ctxt_cross_seg": ctxt_cross_seg,
            "cqa_cross_ids": cqa_cross_ids,
            "topic_cross_ids": topic_cross_ids,
            "user_cross_ids": user_cross_ids,
            "user_flag" : using_user,
            "candidate_prior": float(candidate_prior),
        }


    @staticmethod
    def _process_mentions_for_model(
            mentions,
            top_k,
            xlnet_tokenizer,
            longformer_tokenizer,
            max_seq_length,
            max_desc_length,
            max_q_length,
            debug=False,
            silent=False,
            Training=True,
            blink=True,
            logger=None,
            num_of_cqa=None,
            max_user_q_nums=None,
            max_topic_q_nums=None,
            max_topic_nums=None,
    ):
        processed_mentions = []

        if debug:
            mentions = mentions[:50]

        if silent:
            iter_ = mentions
        else:
            iter_ = tqdm(mentions)

        num_candidate_without_des = 0
        num_total_candidate = 0

        entities = []
        if blink:
            fin = open("../Data/Entity_id_description_blink",
                       "r", encoding='utf-8')
        else:
            fin = open("../Data/Entity_id_description", "r", encoding='utf-8')

        for line in fin.readlines():
            entity = line.split("\t")
            entities.append(entity)

        num_candidate_used_for_training = 0

        for idx, mention in enumerate(iter_):
            if Training:
                if mention["mention_target"] >= top_k:
                    num_candidate_used_for_training += 1
                    continue
            candidates = mention["mention_cand"]

            candidate_features = []

            for candidate in candidates[:top_k]:
                flag = 0
                for entity in entities:
                    if str(candidate[1]).strip() == entity[1].strip():
                        candidate_desc = entity[2]
                        flag = 1
                        break
                if flag == 0:
                    candidate_desc = ""
                    num_candidate_without_des += 1
                num_total_candidate += 1

                candidate_title = candidate[0]
                candidate_obj = CQAEL._get_candidate_tokens_representation(
                    candidate[2],
                    xlnet_tokenizer,
                    longformer_tokenizer,
                    candidate_title,
                    candidate_desc,
                    mention,
                    max_seq_length,
                    max_desc_length,
                    max_q_length,
                    num_of_cqa,
                    max_user_q_nums,
                    max_topic_q_nums,
                    max_topic_nums,
                )
                candidate_features.append(candidate_obj)

            entity_mask = [1] * len(candidate_features) + \
                          [0] * (top_k - len(candidate_features))

            if len(candidate_features) < top_k:
                candidate_title = ""
                candidate_desc = ""
                padding_candidate_obj = CQAEL._get_candidate_tokens_representation(
                    0,
                    xlnet_tokenizer,
                    longformer_tokenizer,
                    candidate_title,
                    candidate_desc,
                    mention,
                    max_seq_length,
                    max_desc_length,
                    max_q_length,
                    num_of_cqa,
                    max_user_q_nums,
                    max_topic_q_nums,
                    max_topic_nums,
                )
                for _ in range(top_k - len(candidate_features)):
                    candidate_features.append(padding_candidate_obj)

            assert len(candidate_features) == top_k
            assert len(entity_mask) == top_k

            label = mention["mention_target"]
            processed_mentions.append(
                {
                    "candidate_features": candidate_features,
                    "label": label,
                    "entity_mask": entity_mask,
                }
            )

        logger.info("total number of candidates is {}".format(
            num_total_candidate))
        logger.info("candidates_without_desc is {}".format(
            num_candidate_without_des))
        logger.info("number of mentions that are filtered for training is {}".format(
            num_candidate_used_for_training))

        all_ctxt_cross_ids = torch.tensor(
            CQAEL._select_field(processed_mentions, "ctxt_cross_ids"),
            dtype=torch.long,
        )
        all_ctxt_cross_mask = torch.tensor(
            CQAEL._select_field(processed_mentions, "ctxt_cross_mask"),
            dtype=torch.long,
        )
        all_ctxt_cross_seg = torch.tensor(
            CQAEL._select_field(processed_mentions, "ctxt_cross_seg"),
            dtype=torch.long,
        )
        all_cqa_cross_ids = torch.tensor(
            CQAEL._select_field(processed_mentions, "cqa_cross_ids"),
            dtype=torch.long,
        )
        all_topic_cross_ids = torch.tensor(
            CQAEL._select_field(processed_mentions, "topic_cross_ids"),
            dtype=torch.long,
        )
        all_user_cross_ids = torch.tensor(
            CQAEL._select_field(processed_mentions, "user_cross_ids"),
            dtype=torch.long,
        )
        all_uesr_flag = torch.tensor(
            CQAEL._select_field(processed_mentions, "user_flag"),
            dtype=torch.long,
        )
        all_candidate_priors = torch.tensor(
            CQAEL._select_field(processed_mentions, "candidate_prior"),
        )
        all_entity_masks = torch.tensor(
            [s["entity_mask"] for s in processed_mentions], dtype=torch.float,
        )
        all_label = torch.tensor(
            [s["label"] for s in processed_mentions], dtype=torch.long,
        )


        data = {
            "all_ctxt_cross_ids": all_ctxt_cross_ids,
            "all_ctxt_cross_mask": all_ctxt_cross_mask,
            "all_ctxt_cross_seg": all_ctxt_cross_seg,
            "all_cqa_cross_ids": all_cqa_cross_ids,
            "all_topic_cross_ids": all_topic_cross_ids,
            "all_user_cross_ids": all_user_cross_ids,
            "all_uesr_flag": all_uesr_flag,
            "all_candidate_priors": all_candidate_priors,
            "all_entity_masks": all_entity_masks,
            "all_label": all_label,
        }

        tensor_data = TensorDataset(
            all_ctxt_cross_ids,
            all_ctxt_cross_mask,
            all_ctxt_cross_seg,

            all_cqa_cross_ids,

            all_candidate_priors,
            all_label,
            all_entity_masks,

            all_topic_cross_ids,
            all_user_cross_ids,
            all_uesr_flag,
        )

        if logger is not None:
            logger.info("all_ctxt_cross_ids shape:{}".format(
                all_ctxt_cross_ids.shape))
            logger.info("all_ctxt_cross_mask shape:{}".format(
                all_ctxt_cross_mask.shape))
            logger.info("all_ctxt_cross_seg shape:{}".format(
                all_ctxt_cross_seg.shape))

            logger.info("all_cqa_cross_ids shape:{}".format(
                all_cqa_cross_ids.shape))
            
            logger.info("all_candidate_priors shape:{}".format(
                all_candidate_priors.shape))
            logger.info("all_entity_masks shape:{}".format(
                all_entity_masks.shape))
            logger.info("all_label shape:{}".format(all_label.shape))

            logger.info("all_topic_cross_ids shape:{}".format(all_topic_cross_ids.shape))
            logger.info("all_user_cross_ids shape:{}".format(all_user_cross_ids.shape))
            logger.info("all_uesr_flag shape:{}".format(all_uesr_flag.shape))
            
                
        return data, tensor_data
