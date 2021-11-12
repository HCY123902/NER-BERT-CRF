import argparse
from re import S
from torch import nn
import torch
from scipy.spatial import distance_matrix
import json
import random

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, dataloader
# from transformers.models.auto.tokenization_auto import tokenizer_class_from_name
from data_pre import NerDataset
from data_pre import CoNLLDataProcessor
from pytorch_pretrained_bert.optimization import BertAdam, warmup_linear
from pytorch_pretrained_bert.tokenization import BertTokenizer
# from data import DialogueDataset
from tqdm import tqdm
import time
# import transformers
import os
from utils import *
import operator
from collections import Counter

class Strategy(object):

    def __init__(self, args, current_model, labeled_data, unlabeled_data, label_map, classifier, labelToIndex={}):
        """
        current_model:
            for training, it is the init model for training
            for retraining, the model has load the ckpt of last iter
        labeled_data: the labeled data list for this iter
        unlabeled_data: the unlabeled data lsit for this iter
        """
        self.args = args
        self.model = current_model.cuda()
        self.labeled_data = labeled_data
        self.unlabeled_data = unlabeled_data
        self.batch_size = args.batch_size
        self.do_lower_case = False
        self.tokenizer = BertTokenizer.from_pretrained(self.args.pretrained_model, do_lower_case=self.do_lower_case)
        self.label_map = label_map
        self.conllProcessor = CoNLLDataProcessor(labelToIndex)

        # Added
        self.classifier_learning_rate0 = 3e-4
        self.classifier_lr0_crf_fc = 8e-5
        self.classifier_weight_decay_finetune = 3e-4  # 0.01
        self.classifier_weight_decay_crf_fc = 5e-6  # 0.005
        self.classifier = classifier.cuda()
        self.classifier_gradient_accumulation_steps = 1
        self.classifier_warmup_proportion = 0.1
        # Prepare optimizer
        param_optimizer = list(self.classifier.named_parameters())

        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        new_param = ['transitions', 'hidden2label.weight', 'hidden2label.bias']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay) \
                        and not any(nd in n for nd in new_param)], 'weight_decay': self.weight_decay_finetune},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay) \
                        and not any(nd in n for nd in new_param)], 'weight_decay': 0.0},
            {'params': [p for n, p in param_optimizer if n in ('transitions', 'hidden2label.weight')] \
                , 'lr': self.lr0_crf_fc, 'weight_decay': self.weight_decay_crf_fc},
            {'params': [p for n, p in param_optimizer if n == 'hidden2label.bias'] \
                , 'lr': self.lr0_crf_fc, 'weight_decay': 0.0}
        ]
        self.classifier_optimizer = BertAdam(optimizer_grouped_parameters, lr=self.learning_rate0, warmup=self.warmup_proportion)
        

    def query(self, n):
        pass

    def get_entropy(self, token_entropy, masks):
        '''
        token_entropy: [sample_size, seq_len]
        masks: [sample_size, seq_len]  [1,1,1,...,0,0,0]
        '''
        EC_score = torch.zeros(len(token_entropy))
        # keep only the valid steps
        for i in range(len(token_entropy)):
            sentence = torch.masked_select(token_entropy[i], masks[i])
            EC_score[i] = sum(sentence) / len(sentence)
        return EC_score

    def update_dataset(self, samps_filter):
        '''
        samps_filter: list
        add the samps into the labeled data
        remove the samps in the unlabeled data
        '''
        # update the labeled_data
        self.labeled_data.extend(samps_filter)
        # update the unlabeled_data
        for samp in samps_filter:
            self.unlabeled_data.remove(samp)

    def evaluate(self, predict_dataloader, batch_size, epoch_th, dataset_name):
        # (model, predict_dataloader, batch_size, epoch_th, dataset_name):
        # print("***** Running prediction *****")
        self.model.eval()
        all_preds = []
        all_labels = []
        total = 0
        correct = 0
        start = time.time()
        with torch.no_grad():
            for batch in predict_dataloader:
                batch = tuple(t.to(self.args.device) for t in batch)
                input_ids, onto_labels, db_labels, input_mask, segment_ids, predict_mask, label_ids, turn_labels = batch
                _, predicted_label_seq_ids, _ = self.model(input_ids, onto_labels, db_labels, segment_ids, input_mask)
                # _, predicted = torch.max(out_scores, -1)
                valid_predicted = torch.masked_select(predicted_label_seq_ids, predict_mask)
                valid_label_ids = torch.masked_select(label_ids, predict_mask)
                all_preds.extend(valid_predicted.tolist())
                all_labels.extend(valid_label_ids.tolist())
                # print(len(valid_label_ids),len(valid_predicted),len(valid_label_ids)==len(valid_predicted))
                total += len(valid_label_ids)
                correct += valid_predicted.eq(valid_label_ids).sum().item()

        test_acc = correct / total
        precision, recall, f1 = defined_f1_score(np.array(all_labels), np.array(all_preds))
        micro = f1_score(all_labels, [t if t > 2 else 3 for t in all_preds], average="micro")
        macro = f1_score(all_labels, [t if t > 2 else 3 for t in all_preds], average="macro")
        spanp, spanr, spanf1 = spanf1_score(all_labels, [t if t > 2 else 3 for t in all_preds])
        end = time.time()
        print(
            'Epoch:%d, Acc:%.2f, Precision: %.2f, Recall: %.2f, F1: %.2f, micro: %.2f, macro: %.2f, span-p: %.2f, span-r: %.2f, spanf1: %.2f on %s, Spend: %.3f minutes for evaluation' \
            % (epoch_th, 100. * test_acc, 100. * precision, 100. * recall, 100. * f1, 100. * micro, 100. * macro,
               100. * spanp, 100. * spanr, 100. * spanf1, dataset_name, (end - start) / 60.0))
        print('--------------------------------------------------------------')
        return test_acc, f1

    def train(self, ind_iter, run_name, n_train_sample, train_dataloader, dev_dataloader, test_dataloader, valid_acc_prev,
              valid_f1_prev, start_epoch, model_path):

        ## trainer configuration
        if run_name in ['all_data', 'init']:
            total_train_epochs = self.args.epoch
            warmup_proportion = 0.1
        else:
            total_train_epochs = 2
            warmup_proportion = 0.0

        # train
        print('training...', total_train_epochs, 'epochs...')
        
        # "The vocabulary file that the BERT model was trained on."
        max_seq_length = 256
        batch_size = 32  # 32
        # "The initial learning rate for Adam."
        
        learning_rate0 = 3e-5
        lr0_crf_fc = 8e-5
        weight_decay_finetune = 1e-5  # 0.01
        weight_decay_crf_fc = 5e-6  # 0.005
        gradient_accumulation_steps = 1
        
        #output_dir = './output/'

        total_train_steps = int(n_train_sample / batch_size / gradient_accumulation_steps * total_train_epochs)

        print("***** Running training *****")
        print("  Num examples = %d" % n_train_sample)
        print("  Batch size = %d" % batch_size)
        print("  Num steps = %d" % total_train_steps)

        self.model.to(self.args.device)

        # Prepare optimizer
        param_optimizer = list(self.model.named_parameters())

        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        new_param = ['transitions', 'hidden2label.weight', 'hidden2label.bias']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay) \
                        and not any(nd in n for nd in new_param)], 'weight_decay': weight_decay_finetune},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay) \
                        and not any(nd in n for nd in new_param)], 'weight_decay': 0.0},
            {'params': [p for n, p in param_optimizer if n in ('transitions', 'hidden2label.weight')] \
                , 'lr': lr0_crf_fc, 'weight_decay': weight_decay_crf_fc},
            {'params': [p for n, p in param_optimizer if n == 'hidden2label.bias'] \
                , 'lr': lr0_crf_fc, 'weight_decay': 0.0}
        ]
        optimizer = BertAdam(optimizer_grouped_parameters, lr=learning_rate0, warmup=warmup_proportion,
                             t_total=total_train_steps)

        # train
        print('training................')
        global_step_th = int(n_train_sample / batch_size / gradient_accumulation_steps * 0)

        for epoch in range(0, total_train_epochs):
            tr_loss = 0
            train_start = time.time()
            self.model.train()
            optimizer.zero_grad()
            # for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
            step = 0
            for batch in tqdm(train_dataloader):
                batch = tuple(t.to(self.args.device) for t in batch)
                input_ids, onto_labels, db_labels, input_mask, segment_ids, predict_mask, label_ids, turn_labels = batch
                neg_log_likelihood = self.model.neg_log_likelihood(input_ids, onto_labels, db_labels, segment_ids,
                                                                   input_mask,
                                                                   label_ids)

                if gradient_accumulation_steps > 1:
                    neg_log_likelihood = neg_log_likelihood / gradient_accumulation_steps

                neg_log_likelihood.backward()

                tr_loss += neg_log_likelihood.item()

                if (step + 1) % gradient_accumulation_steps == 0:
                    # modify learning rate with special warm up BERT uses
                    lr_this_step = learning_rate0 * warmup_linear(global_step_th / total_train_steps, warmup_proportion)
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr_this_step
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step_th += 1

                step += 1

                # print("Epoch:{}-{}/{}, Negative loglikelihood: {} ".format(epoch, step, len(train_dataloader), neg_log_likelihood.item()))

            print('--------------------------------------------------------------')
            print("Epoch:{} completed, Total training's Loss: {}, Spend: {}m".format(epoch, tr_loss,
                                                                                     (
                                                                                             time.time() - train_start) / 60.0))
            valid_acc, valid_f1 = self.evaluate(dev_dataloader, batch_size, epoch, 'Valid_set')
            # Save a checkpoint
            if valid_f1 > valid_f1_prev:
                # model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
                m_path = os.path.join(model_path, run_name)
                if not os.path.exists(m_path):
                    os.makedirs(m_path)
                m_name = "NER_BERT_BILSTM_CRF_KB_"+ str(ind_iter+1)+".pt"
                torch.save({'epoch': epoch, 'model_state': self.model.state_dict(), 'valid_acc': valid_acc,
                            'valid_f1': valid_f1, 'max_seq_length': max_seq_length, 'lower_case': self.do_lower_case},
                           os.path.join(m_path, m_name))
                valid_f1_prev = valid_f1
                valid_acc_prev = valid_acc

        # test
        print('testing................')
        self.evaluate(test_dataloader, batch_size, total_train_epochs - 1, 'Test_set')

        return valid_acc_prev, valid_f1_prev

    def predict_prob(self):
        '''
        return
            probs: [sample_size, seq_len, tags_num]
            masks: [sample_size, seq_len]  [1,1,1,...,0,0,0]
        '''
        self.model.eval()
        # dataloader for unlabeled data
        unlabeled_examples = self.conllProcessor._create_examples(self.unlabeled_data)
        unlabeled_dataset = NerDataset(unlabeled_examples, self.tokenizer, self.label_map, self.args.max_seq_len)
        dataloader = DataLoader(unlabeled_dataset, batch_size=self.batch_size, shuffle=False, collate_fn=NerDataset.padselect)

        probs = torch.zeros((len(unlabeled_dataset), self.args.max_seq_len, len(self.label_map)))
        masks = torch.zeros((len(unlabeled_dataset), self.args.max_seq_len))
        with torch.no_grad():
            i = 0
            for batch in tqdm(dataloader):
                batch = tuple(t.to(self.args.device) for t in batch)
                input_ids, onto_labels, db_labels, input_mask, segment_ids, predict_mask, label_ids, turn_labels = batch
                _, predicted_label_seq_ids, score_l = self.model(input_ids, onto_labels, db_labels, segment_ids,
                                                                 input_mask)
                prob = F.softmax(score_l, dim=-1)
                #size of prob, mask torch.Size([16, 74, 6]) torch.Size([16, 74])
                #print("size of prob, mask", prob.size(), predict_mask.size())
                mask = predict_mask

                if (i + 1) * self.batch_size <= len(unlabeled_dataset):
                    probs[self.batch_size * i:self.batch_size * (i + 1)] = prob.cpu()
                    masks[self.batch_size * i:self.batch_size * (i + 1)] = mask.cpu()
                else:
                    probs[self.batch_size * i:] = prob.cpu()
                    masks[self.batch_size * i:] = mask.cpu()
                i += 1
        return probs, masks

    def predict_prob_dropout_split(self, n_trial):
        # n_trial : the number of repeat for dropout
        self.model.train()  ## dropout as variational Bayesian approximation
        # dataloader for unlabeled data
        unlabeled_examples = self.conllProcessor._create_examples(self.unlabeled_data)
        unlabeled_dataset = NerDataset(unlabeled_examples, self.tokenizer, self.label_map, self.args.max_seq_len)
        dataloader = DataLoader(unlabeled_dataset, batch_size=self.batch_size, shuffle=False, collate_fn=NerDataset.padselect)
        probs = torch.zeros((n_trial, len(unlabeled_dataset), self.args.max_seq_len, len(self.label_map)))
        masks = torch.zeros((len(unlabeled_dataset), self.args.max_seq_len))
        for k in range(n_trial):
            seed_torch(seed=k)
            with torch.no_grad():
                i = 0
                for batch in tqdm(dataloader):
                    batch = tuple(t.to(self.args.device) for t in batch)
                    input_ids, onto_labels, db_labels, input_mask, segment_ids, predict_mask, label_ids, turn_labels = batch
                    _, predicted_label_seq_ids, score_l = self.model(input_ids, onto_labels, db_labels, segment_ids,
                                                                     input_mask)
                    prob = F.softmax(score_l, dim=-1)
                    #print("size of prob, mask", prob.size(), predict_mask.size())
                    mask = predict_mask

                    if (i + 1) * self.batch_size <= len(unlabeled_dataset):
                        probs[k][self.batch_size * i:self.batch_size * (i + 1)] = prob.cpu()
                        if k == 0:
                            masks[self.batch_size * i:self.batch_size * (i + 1)] = mask.cpu()
                    else:
                        probs[k][self.batch_size * i:] = prob.cpu()
                        if k == 0:
                            masks[self.batch_size * i:] = mask.cpu()
                    i += 1
        return probs, masks
    
    def evaluate_turn_label(self, predict_dataloader, batch_size, epoch_th, dataset_name):
        # print("***** Running prediction *****")
        self.classifier.eval()
        all_preds = []
        all_turn_labels = []
        total = 0
        # Adjusted
        # correct = 0
        correct_at_1 = 0
        correct_at_2 = 0
        correct_at_3 = 0

        start = time.time()
        with torch.no_grad():
            for batch in predict_dataloader:
                batch = tuple(t.to(self.args.device) for t in batch)
                input_ids, onto_labels, db_labels, input_mask, segment_ids, predict_mask, label_ids, turn_labels = batch
                predicted_turn_labels = self.classifier(input_ids, onto_labels, db_labels, segment_ids, input_mask)
                # _, predicted = torch.max(out_scores, -1)
                # valid_predicted = torch.masked_select(predicted_label_seq_ids, predict_mask)
                # valid_label_ids = torch.masked_select(label_ids, predict_mask)
                predicted_turn_labels = predicted_turn_labels.tolist()
                turn_labels = turn_labels.tolist()

                all_preds.extend(predicted_turn_labels)
                all_turn_labels.extend(turn_labels)
                # print(len(valid_label_ids),len(valid_predicted),len(valid_label_ids)==len(valid_predicted))
                total += len(turn_labels)
                # correct += valid_predicted.eq().sum().item()
                correct_at_1 = correct_at_1 + len([1 for p, t in zip(predicted_turn_labels, turn_labels) if p[0] == t])
                correct_at_2 = correct_at_2 + len([1 for p, t in zip(predicted_turn_labels, turn_labels) if t in p[:2]])
                correct_at_3 = correct_at_3 + len([1 for p, t in zip(predicted_turn_labels, turn_labels) if t in p[:3]])

        # Adjusted
        test_acc = float(correct_at_1) / float(total)
        precision = float(correct_at_1) / float(total)
        recall_at_1 = float(correct_at_1) / float(total)
        recall_at_2 = float(correct_at_2) / float(total)
        recall_at_3 = float(correct_at_3) / float(total)
        try:
            f1 = 2 * precision * recall_at_1 / (precision + recall_at_1)
        except ZeroDivisionError:
            f1 = 1.0
        
        # micro = f1_score(all_labels, [t if t > 2 else 3 for t in all_preds], average="micro")
        # macro = f1_score(all_labels, [t if t > 2 else 3 for t in all_preds], average="macro")
        # print("before span")
        # spanp, spanr, spanf1 = spanf1_score(all_labels, [t if t > 2 else 3 for t in all_preds])
        # print("after span")
        end = time.time()
        print('Epoch:%d, Acc:%.6f, Precision: %.6f, Recall_at_1: %.6f, Recall_at_2: %.6f, Recall_at_3: %.6f, F1: %.6f, Spend: %.3f minutes for evaluation on the set %s' % (epoch_th, 100. * test_acc, 100. * precision, 100. * recall_at_1, 100. * recall_at_2, 100. * recall_at_3, 100. * f1, (end - start) / 60.0, dataset_name))
        print('--------------------------------------------------------------')
        return test_acc, f1

    def train_classifier(self, ind_iter, n_train_sample, train_dataloader, dev_dataloader, valid_acc_prev,
              valid_f1_prev, start_epoch, model_path, number_of_epochs):

        ## trainer configuration
        total_train_epochs = number_of_epochs

        # train
        print('training...', total_train_epochs, 'epochs...')
        
        # "The vocabulary file that the BERT model was trained on."
        max_seq_length = 256
        batch_size = self.args.classifier_batch_size  # 128
        # "The initial learning rate for Adam."
    
        
        #output_dir = './output/'

        total_train_steps = int(n_train_sample / batch_size / self.classifier_gradient_accumulation_steps * total_train_epochs)

        print("***** Running classifier training *****")
        print("  Num examples = %d" % n_train_sample)
        print("  Batch size = %d" % batch_size)
        print("  Num steps = %d" % total_train_steps)

        self.model.to(self.args.device)


        # train
        print('training................')
        global_step_th = int(n_train_sample / batch_size / self.classifier_gradient_accumulation_steps * 0)

        for epoch in range(0, total_train_epochs):
            tr_loss = 0
            train_start = time.time()
            self.model.train()
            self.classifier_optimizer.zero_grad()
            # for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
            step = 0
            for batch in tqdm(train_dataloader):
                batch = tuple(t.to(self.args.device) for t in batch)
                input_ids, onto_labels, db_labels, input_mask, segment_ids, predict_mask, label_ids, turn_labels = batch

                cross_entropy_loss = self.classifier.cross_entropy_loss(input_ids, onto_labels, db_labels, segment_ids, input_mask,
                                                            turn_labels)

                if self.classifier_gradient_accumulation_steps > 1:
                    cross_entropy_loss = cross_entropy_loss / self.classifier_gradient_accumulation_steps
                cross_entropy_loss.backward()
                tr_loss += cross_entropy_loss.item()

                if (step + 1) % self.classifier_gradient_accumulation_steps == 0:
                    # modify learning rate with special warm up BERT uses
                    lr_this_step = self.classifier_learning_rate0 * warmup_linear(global_step_th / total_train_steps, self.classifier_warmup_proportion)
                    for param_group in self.classifier_optimizer.param_groups:
                        param_group['lr'] = lr_this_step
                    self.classifier_optimizer.step()
                    self.classifier_optimizer.zero_grad()
                    global_step_th += 1

                step += 1

                # print("Epoch:{}-{}/{}, Negative loglikelihood: {} ".format(epoch, step, len(train_dataloader), neg_log_likelihood.item()))

            print('--------------------------------------------------------------')
            print("Epoch:{} completed, Total training's Loss: {}, Spend: {}m".format(epoch, tr_loss,
                                                                                     (
                                                                                             time.time() - train_start) / 60.0))
            valid_acc, valid_f1 = self.evaluate_turn_label(dev_dataloader, batch_size, epoch, 'Valid_set')
            # Save a checkpoint
            if valid_f1 > valid_f1_prev:
                # model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
                m_path = model_path
                if not os.path.exists(m_path):
                    os.makedirs(m_path)
                m_name = "NER_BERT_BILSTM_CRF_KB_CLASSIFIER_"+ str(ind_iter+1)+".pt"
                torch.save({'epoch': epoch, 'model_state': self.model.state_dict(), 'valid_acc': valid_acc,
                            'valid_f1': valid_f1, 'max_seq_length': max_seq_length, 'lower_case': self.do_lower_case},
                           os.path.join(m_path, m_name))
                valid_f1_prev = valid_f1
                valid_acc_prev = valid_acc

        return valid_acc_prev, valid_f1_prev

    def classify_labeled_and_unlabeled_set(self, labeled_loader, unlabeled_loader):
        labeled_map = {}
        unlabeled_map = {}
        labeled_prediction = []
        unlabeled_prediction = []
        with torch.no_grad():
            for batch in labeled_loader:
                batch = tuple(t.to(self.args.device) for t in batch)
                input_ids, onto_labels, db_labels, input_mask, segment_ids, predict_mask, label_ids, turn_labels = batch
                predicted_turn_labels = self.classifier(input_ids, onto_labels, db_labels, segment_ids, input_mask)
                # _, predicted = torch.max(out_scores, -1)
                # valid_predicted = torch.masked_select(predicted_label_seq_ids, predict_mask)
                # valid_label_ids = torch.masked_select(label_ids, predict_mask)
                predicted_turn_labels = predicted_turn_labels.tolist()

                labeled_prediction.extend(predicted_turn_labels)
            
            for batch in unlabeled_loader:
                batch = tuple(t.to(self.args.device) for t in batch)
                input_ids, onto_labels, db_labels, input_mask, segment_ids, predict_mask, label_ids, turn_labels = batch
                predicted_turn_labels = self.classifier(input_ids, onto_labels, db_labels, segment_ids, input_mask)
                # _, predicted = torch.max(out_scores, -1)
                # valid_predicted = torch.masked_select(predicted_label_seq_ids, predict_mask)
                # valid_label_ids = torch.masked_select(label_ids, predict_mask)
                predicted_turn_labels = predicted_turn_labels.tolist()

                unlabeled_prediction.extend(predicted_turn_labels)

        for i, prediction in enumerate(labeled_prediction):
            labeled_map.setdefault(int(prediction), []).append(i)

        for i, prediction in enumerate(unlabeled_prediction):
            unlabeled_map.setdefault(int(prediction), []).append(i)

        return labeled_map, unlabeled_map, unlabeled_prediction

    def sample_with_ranking(self, n, rank, labeled_loader, unlabeled_loader):
        # Adjusted
        selected_data = []
        labeled_map, unlabeled_map, unlabeled_prediction = self.classify_labeled_and_unlabeled_set(labeled_loader, unlabeled_loader)

        if len(unlabeled_prediction) <= n:
            _, top_n_index = torch.topk(rank, n, largest=True)  # 从大到小
            top_n_index = top_n_index.numpy().tolist()

            for index in top_n_index:
                selected_data.append(self.unlabeled_data[index])
            return selected_data
        else:
            class_number = len(labeled_map) + len(unlabeled_map)
            print("Number of classes in prediction is {}".format(class_number))
            sample_number = np.ceil((float(len(self.labeled_data)) + float(n)) / float(class_number))
            print("Expected sample number in each class is {}".format(sample_number))


            _, top_n_index = torch.topk(rank, len(self.unlabeled_data), largest=True)  # 从大到小
            top_n_index = top_n_index.numpy().tolist()

            selected_class_count = {}

            for index in top_n_index:
                class_count = selected_class_count.get(unlabeled_prediction[index], 0)
                if class_count < sample_number:
                    selected_data.append(self.unlabeled_data[index])
                    selected_class_count[unlabeled_prediction[index]] = class_count + 1
                if len(selected_data) >= n:
                    break
                
            return selected_data[:n]

class RandomSampling(Strategy):  ## return selected samples (n pieces) for current iteration
    def __init__(self, args, current_model, labeled_data, unlabeled_data, label_map, classifier, labelToIndex={}):
        super().__init__(args, current_model, labeled_data, unlabeled_data, label_map, classifier, labelToIndex=labelToIndex)

    def query(self, n, labeled_loader, unlabeled_loader):
        # return random.sample(self.unlabeled_data, n)

        labeled_map, unlabeled_map, unlabeled_prediction = self.classify_labeled_and_unlabeled_set(labeled_loader, unlabeled_loader)

        if len(unlabeled_prediction) <= n:
            return self.unlabeled_data[:n]

        class_number = len(labeled_map) + len(unlabeled_map)
        print("Number of classes in prediction is {}".format(class_number))
        sample_number = np.ceil((float(len(self.labeled_data)) + float(n)) / float(class_number))
        print("Expected sample number in each class is {}".format(sample_number))

        result = []

        while len(result) < n:
            for key in list(labeled_map.keys()) + list(unlabeled_map.keys()):
                current = len(labeled_map.get(key, []))
                potential = len(unlabeled_map.get(key, []))
                if potential == 0:
                    continue
                if current == 0:
                    result.extend([self.unlabeled_data[i] for i in unlabeled_map[key][:sample_number]])
                    unlabeled_map[key] = unlabeled_map[key][sample_number:]
                elif current >= sample_number:
                    continue
                else:
                    result.extend([self.unlabeled_data[i] for i in unlabeled_map[key][:sample_number - current]])
                    unlabeled_map[key] = unlabeled_map[key][sample_number - current:]

        return result[:n]


class MarginSampling(Strategy):

    def __init__(self, args, current_model, labeled_data, unlabeled_data, label_map, classifier, labelToIndex={}):
        super().__init__(args, current_model, labeled_data, unlabeled_data, label_map, classifier, labelToIndex=labelToIndex)

    def query(self, n, labeled_loader, unlabeled_loader):
        probs, masks = self.predict_prob()
        '''
        probs: [sample_size, seq_len, tags_num]
        masks: [sample_size, seq_len]  [1,1,1,...,0,0,0]
        '''
        masks = (masks == 1)
        probs_sorted, idxs = probs.sort(descending=True)
        U = probs_sorted[:, :, 0] - probs_sorted[:, :, 1]
        U_valid_data = torch.zeros(len(U))
        for i in range(len(U)):
            U_valid_data[i] = torch.masked_select(U[i], masks[i]).sum()
        final_score = U_valid_data
        if not self.args.turn_level:
            # aggregate the samples belonging to the same dial
            len_dial = len(self.dial_len_unlabel_sum) - 1
            assert len_dial == len(self.unlabeled_data), 'Error in sampling unlabled data'
            dial_score_avg = torch.zeros(len_dial)
            for i in range(len_dial):
                dial_score = U_valid_data[self.dial_len_unlabel_sum[i]:self.dial_len_unlabel_sum[i + 1]]
                dial_score_avg[i] = dial_score.mean()
            final_score = dial_score_avg

        selected_data = []
        new_indices = final_score.sort()[1][:n]
        for index in new_indices:
            selected_data.append(self.unlabeled_data[index])
        return selected_data


class EntropySampling(Strategy):

    def __init__(self, args, current_model, labeled_data, unlabeled_data, label_map, classifier, labelToIndex={}):
        super().__init__(args, current_model, labeled_data, unlabeled_data, label_map, classifier, labelToIndex=labelToIndex)

    def query(self, n, labeled_loader, unlabeled_loader):
        probs, masks = self.predict_prob()
        '''
        probs: [sample_size, seq_len, tags_num]
        masks: [sample_size, seq_len]  [1,1,1,...,0,0,0]
        '''
        masks = (masks == 1)
        log_probs = torch.log(probs)
        token_entropy = -(probs * log_probs).sum(-1)  # [sample_size, seq_len]
        assert token_entropy.shape == masks.shape
        EC_score = self.get_entropy(token_entropy, masks)
        final_score = EC_score
        if not self.args.turn_level:
            # aggregate the samples belonging to the same dial
            len_dial = len(self.dial_len_unlabel_sum) - 1
            assert len_dial == len(self.unlabeled_data), 'Error in sampling unlabled data'
            dial_score_avg = torch.zeros(len_dial)
            for i in range(len_dial):
                dial_score = EC_score[self.dial_len_unlabel_sum[i]:self.dial_len_unlabel_sum[i + 1]]
                dial_score_avg[i] = dial_score.mean()
            final_score = dial_score_avg

        # Adjusted
        # _, top_n_index = torch.topk(final_score, n, largest=True)  # 从大到小
        # top_n_index = top_n_index.numpy().tolist()
        # selected_data = []
        # for index in top_n_index:
        #     selected_data.append(self.unlabeled_data[index])
        # return selected_data

        return self.sample_with_ranking(n, final_score, labeled_loader, unlabeled_loader)

        


class BALDDropout2(Strategy):
    def __init__(self, args, current_model, labeled_data, unlabeled_data, label_map, classifier, labelToIndex={}):
        super().__init__(args, current_model, labeled_data, unlabeled_data, label_map, classifier, labelToIndex=labelToIndex)
        self.n_trail = 2

    def query(self, n, labeled_loader, unlabeled_loader):
        viterbi_sequences, scores = {}, {}
        count = []
        print('get the dropout output')
        probs, masks = self.predict_prob_dropout_split(self.n_trail)
        '''
        probs: [n_trail, sample_size, seq_len, tags_num]
        masks: [sample_size, seq_len]  [1,1,1,...,0,0,0]
        '''
        print('done')
        masks = (masks == 1)
        for prob in probs:
            # [sample_size, seq_len, tags_num]
            for sent_id, (logits, mask) in enumerate(zip(prob, masks)):
                # logit = torch.masked_select(logits, mask)
                sorted, indices = torch.sort(logits, descending=True)
                # max_prob = torch.masked_select(sorted[:,0],mask)
                max_ind = torch.masked_select(indices[:, 0], mask)
                if sent_id in viterbi_sequences.keys():
                    viterbi_sequences[sent_id] += [max_ind.tolist()]
                    # scores[sent_id] += [max_prob.tolist()]
                else:
                    viterbi_sequences[sent_id] = [max_ind.tolist()]
                    # scores[sent_id] = [max_prob.tolist()]
        count += [Counter(tuple(x) for x in value) for value in viterbi_sequences.values()]
        scores = [1 - (max(dict.items(), key=operator.itemgetter(1))[1]) / self.n_trail for dict in count]
        
        # Adjusted
        # _, top_n_index = torch.topk(torch.Tensor(scores), n, largest=True)  # 从大到小
        # top_n_index = top_n_index.numpy().tolist()
        # selected_data = []
        # for index in top_n_index:
        #     selected_data.append(self.unlabeled_data[index])
        # return selected_data

        return self.sample_with_ranking(n, torch.Tensor(scores), labeled_loader, unlabeled_loader)


class BALDDropout(Strategy):
    def __init__(self, args, current_model, labeled_data, unlabeled_data, label_map, classifier, labelToIndex={}):
        super().__init__(args, current_model, labeled_data, unlabeled_data, label_map, classifier, labelToIndex=labelToIndex)
        self.n_trail = 5

    def query(self, n, labeled_loader, unlabeled_loader):
        probs, masks = self.predict_prob_dropout_split(self.n_trail)
        '''
        probs: [n_trail, sample_size, seq_len, tags_num]
        masks: [sample_size, seq_len]  [1,1,1,...,0,0,0]
        '''
        masks = (masks == 1)
        log_probs = torch.log(probs)
        token_entropy = (probs * log_probs).sum(-1)  # [n_trail,sample_size, seq_len]
        # entropy and then avg on n_trail [n_trail, sample_size]
        entropy_avg = torch.zeros([self.n_trail, token_entropy.size(1)])
        for i, token_entropy_i in enumerate(token_entropy):
            entropy_avg[i] = self.get_entropy(token_entropy_i, masks)
        entropy_avg = torch.mean(entropy_avg, dim=0)

        avg_probs = torch.mean(probs, 0)
        avg_entropy = (avg_probs * torch.log(avg_probs)).sum(-1)
        avg_entropy = self.get_entropy(avg_entropy, masks)
        U = entropy_avg - avg_entropy
        final_score = U
        if not self.args.turn_level:
            # aggregate the samples belonging to the same dial
            len_dial = len(self.dial_len_unlabel_sum) - 1
            assert len_dial == len(self.unlabeled_data), 'Error in sampling unlabled data'
            dial_score_avg = torch.zeros(len_dial)
            for i in range(len_dial):
                dial_score = U[self.dial_len_unlabel_sum[i]:self.dial_len_unlabel_sum[i + 1]]
                dial_score_avg[i] = dial_score.mean()
            final_score = dial_score_avg

        # Adjusted
        # _, top_n_index = torch.topk(final_score, n, largest=True)  # 从大到小
        # top_n_index = top_n_index.numpy().tolist()
        # selected_data = []
        # for index in top_n_index:
        #     selected_data.append(self.unlabeled_data[index])
        # return selected_data

        return self.sample_with_ranking(n, final_score, labeled_loader, unlabeled_loader)