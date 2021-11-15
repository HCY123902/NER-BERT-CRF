import time
import importlib
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.autograd as autograd
import torch.optim as optim

from torch.utils.data.distributed import DistributedSampler
from torch.utils import data

from tqdm import tqdm, trange
import collections

from pytorch_pretrained_bert.modeling import BertModel, BertForTokenClassification, BertLayerNorm
import pickle
# from pytorch_pretrained_bert.optimization import BertAdam, WarmupLinearSchedule
from pytorch_pretrained_bert.optimization import BertAdam, warmup_linear
from pytorch_pretrained_bert.tokenization import BertTokenizer

import json

class BERT_BILSTM_CRF_KB_NER_CLASSIFIER(nn.Module):

    def __init__(self, bert_model, start_label_id, stop_label_id, num_labels, max_seq_length, batch_size, device, turn_label_size, gradient=0):
        super(BERT_BILSTM_CRF_KB_NER_CLASSIFIER, self).__init__()
        self.hidden_size = 768
        self.start_label_id = start_label_id
        self.stop_label_id = stop_label_id
        self.num_labels = num_labels
        # self.max_seq_length = max_seq_length
        self.batch_size = batch_size
        self.device = device

        # use pretrainded BertModel
        self.bert = bert_model

        # Added
        if gradient != 1:
            for name, param in self.bert.named_parameters():
                if 'classifier' not in name: # classifier layer
                    param.requires_grad = False

        self.dropout = torch.nn.Dropout(0.2)

        self.onto_embed = torch.nn.Embedding(2, 5)
        self.db_embed = torch.nn.Embedding(2, 5)

        # Maps the output of the bert into label space.
        self.rnn = nn.LSTM(bidirectional=True, num_layers=2, input_size=778, hidden_size=778 // 2, batch_first=True)
        # self.hidden2label = nn.Linear(self.hidden_size + 10, self.num_labels)

        # Matrix of transition parameters.  Entry i,j is the score of transitioning *to* i *from* j.
        # self.transitions = nn.Parameter(
        #     torch.randn(self.num_labels, self.num_labels))

        # These two statements enforce the constraint that we never transfer *to* the start tag(or label),
        # and we never transfer *from* the stop label (the model would probably learn this anyway,
        # so this enforcement is likely unimportant)
        # self.transitions.data[start_label_id, :] = -10000
        # self.transitions.data[:, stop_label_id] = -10000

        # nn.init.xavier_uniform_(self.hidden2label.weight)
        # nn.init.constant_(self.hidden2label.bias, 0.0)
        # self.apply(self.init_bert_weights)

        # Added
        
        self.mlp_turn_classifier = nn.Sequential(
                # batch_size, max_contiext_len, max_context_len, hidden_size * 4 -> batch_size, max_context_len, max_context_len, hidden_size
                nn.Linear(778 // 2, turn_label_size),
                nn.BatchNorm1d(turn_label_size, eps=1e-05, momentum=0.1),
                nn.ReLU(),
                nn.Linear(turn_label_size, turn_label_size),
                nn.BatchNorm1d(turn_label_size, eps=1e-05, momentum=0.1),
                nn.ReLU(),
                # batch_size, max_context_len, max_context_len, hidden_size -> batch_size, max_context_len, max_context_len, 1
                nn.Linear(turn_label_size, turn_label_size),
            )

        self.criterion = nn.CrossEntropyLoss()
        

    def init_bert_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, BertLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    # def _forward_alg(self, feats):
    #     '''
    #     this also called alpha-recursion or forward recursion, to calculate log_prob of all barX
    #     '''

    #     # T = self.max_seq_length
    #     T = feats.shape[1]
    #     batch_size = feats.shape[0]

    #     # alpha_recursion,forward, alpha(zt)=p(zt,bar_x_1:t)
    #     log_alpha = torch.Tensor(batch_size, 1, self.num_labels).fill_(-10000.).to(self.device)
    #     # normal_alpha_0 : alpha[0]=Ot[0]*self.PIs
    #     # self.start_label has all of the score. it is log,0 is p=1
    #     log_alpha[:, 0, self.start_label_id] = 0

    #     # feats: sentances -> word embedding -> lstm -> MLP -> feats
    #     # feats is the probability of emission, feat.shape=(1,tag_size)
    #     for t in range(1, T):
    #         log_alpha = (log_sum_exp_batch(self.transitions + log_alpha, axis=-1) + feats[:, t]).unsqueeze(1)

    #     # log_prob of all barX
    #     log_prob_all_barX = log_sum_exp_batch(log_alpha)
    #     return log_prob_all_barX

    def _get_bert_features(self, input_ids, onto_labels, db_labels, segment_ids, input_mask):
        '''
        sentances -> word embedding -> lstm -> MLP -> feats
        '''
        bert_seq_out, _ = self.bert(input_ids, token_type_ids=segment_ids, attention_mask=input_mask,
                                    output_all_encoded_layers=False)

        # Adjusted
        new_seq_out = torch.cat((bert_seq_out, self.onto_embed(onto_labels), self.db_embed(db_labels)), 2)
        new_seq_out = self.dropout(new_seq_out)

        # h_n: 4, batch_size, hidden_size
        _, (h_n, c_n) = self.rnn(new_seq_out)
        # batch, hidden_size
        hidden = torch.sum(h_n, 0)
        return hidden

    # def _score_sentence(self, feats, label_ids):
    #     '''
    #     Gives the score of a provided label sequence
    #     p(X=w1:t,Zt=tag1:t)=...p(Zt=tag_t|Zt-1=tag_t-1)p(xt|Zt=tag_t)...
    #     '''

    #     # T = self.max_seq_length
    #     T = feats.shape[1]
    #     batch_size = feats.shape[0]

    #     batch_transitions = self.transitions.expand(batch_size, self.num_labels, self.num_labels)
    #     batch_transitions = batch_transitions.flatten(1)

    #     score = torch.zeros((feats.shape[0], 1)).to(device)
    #     # the 0th node is start_label->start_word,the probability of them=1. so t begin with 1.
    #     for t in range(1, T):
    #         score = score + \
    #                 batch_transitions.gather(-1, (label_ids[:, t] * self.num_labels + label_ids[:, t - 1]).view(-1, 1)) \
    #                 + feats[:, t].gather(-1, label_ids[:, t].view(-1, 1)).view(-1, 1)
    #     return score

    # def _viterbi_decode(self, feats):
    #     '''
    #     Max-Product Algorithm or viterbi algorithm, argmax(p(z_0:t|x_0:t))
    #     '''

    #     # T = self.max_seq_length
    #     T = feats.shape[1]
    #     batch_size = feats.shape[0]

    #     # batch_transitions=self.transitions.expand(batch_size,self.num_labels,self.num_labels)

    #     log_delta = torch.Tensor(batch_size, 1, self.num_labels).fill_(-10000.).to(self.device)
    #     log_delta[:, 0, self.start_label_id] = 0

    #     # psi is for the vaule of the last latent that make P(this_latent) maximum.
    #     psi = torch.zeros((batch_size, T, self.num_labels), dtype=torch.long).to(self.device)  # psi[0]=0000 useless
    #     for t in range(1, T):
    #         # delta[t][k]=max_z1:t-1( p(x1,x2,...,xt,z1,z2,...,zt-1,zt=k|theta) )
    #         # delta[t] is the max prob of the path from  z_t-1 to z_t[k]
    #         log_delta, psi[:, t] = torch.max(self.transitions + log_delta, -1)
    #         # psi[t][k]=argmax_z1:t-1( p(x1,x2,...,xt,z1,z2,...,zt-1,zt=k|theta) )
    #         # psi[t][k] is the path choosed from z_t-1 to z_t[k],the value is the z_state(is k) index of z_t-1
    #         log_delta = (log_delta + feats[:, t]).unsqueeze(1)

    #     # trace back
    #     path = torch.zeros((batch_size, T), dtype=torch.long).to(self.device)

    #     # max p(z1:t,all_x|theta)
    #     max_logLL_allz_allx, path[:, -1] = torch.max(log_delta.squeeze(), -1)

    #     for t in range(T - 2, -1, -1):
    #         # choose the state of z_t according the state choosed of z_t+1.
    #         path[:, t] = psi[:, t + 1].gather(-1, path[:, t + 1].view(-1, 1)).squeeze()

    #     return max_logLL_allz_allx, path

    # def neg_log_likelihood(self, input_ids, onto_labels, db_labels, segment_ids, input_mask, label_ids):
    #     bert_feats = self._get_bert_features(input_ids, onto_labels, db_labels, segment_ids, input_mask)
    #     forward_score = self._forward_alg(bert_feats)
    #     # p(X=w1:t,Zt=tag1:t)=...p(Zt=tag_t|Zt-1=tag_t-1)p(xt|Zt=tag_t)...
    #     gold_score = self._score_sentence(bert_feats, label_ids)
    #     # - log[ p(X=w1:t,Zt=tag1:t)/p(X=w1:t) ] = - log[ p(Zt=tag1:t|X=w1:t) ]
    #     return torch.mean(forward_score - gold_score)

    # this forward is just for predict, not for train
    # dont confuse this with _forward_alg above.
    def forward(self, input_ids, onto_labels, db_labels, segment_ids, input_mask):
        # Get the emission scores from the BiLSTM
        bert_feats = self._get_bert_features(input_ids, onto_labels, db_labels, segment_ids, input_mask)

        turn_label_prediction = self.mlp_turn_classifier(bert_feats)
        _, turn_labels = torch.topk(turn_label_prediction, 3, dim=1)
        return turn_labels


    # Added for turn label prediction
    def cross_entropy_loss(self, input_ids, onto_labels, db_labels, segment_ids, input_mask, turn_label_target):
        bert_feats = self._get_bert_features(input_ids, onto_labels, db_labels, segment_ids, input_mask)

        # batch_size, turn_label_size
        turn_label_prediction = self.mlp_turn_classifier(bert_feats)

        return self.criterion(turn_label_prediction, turn_label_target)

