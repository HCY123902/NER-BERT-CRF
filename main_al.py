
'''
This section of code is partly based on ej0cl6/deep-active-learning 
and dsgissin/DiscriminativeActiveLearning from github.
'''
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '6'

import argparse
from re import S
from classifier import BERT_BILSTM_CRF_KB_NER_CLASSIFIER
from torch import nn
import torch
from scipy.spatial import distance_matrix
import json
import random

from strategy_goal import *
import time
from copy import deepcopy
from tqdm import tqdm, trange
import collections
from model import *
from tqdm import tqdm
from pytorch_pretrained_bert.modeling import BertModel, BertForTokenClassification, BertLayerNorm
import pickle
from pytorch_pretrained_bert.optimization import BertAdam, warmup_linear
from pytorch_pretrained_bert.tokenization import BertTokenizer
from param_parser import parameter_parser
import pickle
from utils import *
import sys
from torch.utils import data

if __name__ == '__main__':
    args = parameter_parser()

    seed_torch(seed=44)
    log_path = os.path.join("log_print","log")
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    print('Python version ', sys.version)
    print('PyTorch version ', torch.__version__)
    print('Current dir:', os.getcwd())
    cuda_yes = torch.cuda.is_available()
    print('Cuda is available?', cuda_yes)
    device = torch.device("cuda:0" if cuda_yes else "cpu")
    args.device = device
    print('Device:', device)

    max_seq_length = 256
    do_lower_case = False
    data_dir = os.path.join('./', 'mmconv_data/')
    tokenizer = BertTokenizer.from_pretrained(args.pretrained_model, do_lower_case=do_lower_case)

    # Added
    with open("./turn_label.json", "r", encoding="utf-8") as t:
        turn_label_map = json.load(t)

    labelToIndex = turn_label_map["labelToIndex"]
    turn_label_size = len(labelToIndex)

    # Adjusted
    conllProcessor = CoNLLDataProcessor(labelToIndex)

    label_list = conllProcessor.get_labels()
    label_map = conllProcessor.get_label_map()
    print('len_label_list: ', len(label_list))
    print('label_map: ', label_map)
    all_train_examples = conllProcessor.get_train_examples(data_dir)  # just for # of all training sample
    dev_examples = conllProcessor.get_dev_examples(data_dir)
    test_examples = conllProcessor.get_test_examples(data_dir)
    #test_examples,_ = conllProcessor.get_train_examples_ratio(data_dir,0.0002)
    dev_dataset = NerDataset(dev_examples, tokenizer, label_map, max_seq_length)
    test_dataset = NerDataset(test_examples, tokenizer, label_map, max_seq_length)

    # 1. init training
    run_name = "init"
    method_name = args.coarse_sampling
    # log = open(os.path.join(log_path, run_name+'_'+ args.coarse_sampling +".txt"), "w")
    # sys.stdout = log
    labeled_data, unlabeled_data = conllProcessor.get_train_data_list_ratio(data_dir, args.init_data_ratio)
    labeled_dataset = conllProcessor._create_examples(labeled_data)
    init_train_dataset = NerDataset(labeled_dataset, tokenizer, label_map, max_seq_length)

    train_dataloader = data.DataLoader(dataset=init_train_dataset,
                                       batch_size=args.batch_size,
                                       shuffle=True,
                                       num_workers=4,
                                       collate_fn=NerDataset.pad)

    dev_dataloader = data.DataLoader(dataset=dev_dataset,
                                     batch_size=args.batch_size,
                                     shuffle=False,
                                     num_workers=4,
                                     collate_fn=NerDataset.pad)

    test_dataloader = data.DataLoader(dataset=test_dataset,
                                      batch_size=args.batch_size,
                                      shuffle=False,
                                      num_workers=4,
                                      collate_fn=NerDataset.pad)

    # Added
    classifier_train_dataset = NerDataset(all_train_examples, tokenizer, label_map, max_seq_length)
    classifier_train_dataloader = data.DataLoader(dataset=classifier_train_dataset,
                                       batch_size=args.classifier_batch_size,
                                       shuffle=True,
                                       num_workers=4,
                                       collate_fn=NerDataset.pad)

    classifier_dev_dataloader = data.DataLoader(dataset=dev_dataset,
                                       batch_size=args.classifier_batch_size,
                                       shuffle=False,
                                       num_workers=4,
                                       collate_fn=NerDataset.pad)


    model_path = os.path.join('checkpoint', method_name)
    if not os.path.exists(model_path):
        os.makedirs(model_path)  

    start = time.time()
    start_label_id = conllProcessor.get_start_label_id()
    stop_label_id = conllProcessor.get_stop_label_id()

    tagging_model = BERT_BILSTM_CRF_KB_NER( BertModel.from_pretrained(args.pretrained_model), start_label_id, stop_label_id, len(label_list), args.max_seq_len, args.batch_size, device)

    classifier = BERT_BILSTM_CRF_KB_NER_CLASSIFIER(BertModel.from_pretrained(args.pretrained_model), start_label_id, stop_label_id, len(label_list), args.max_seq_len, args.classifier_batch_size, device, turn_label_size)

    model_exact_path = os.path.join('checkpoint/'+ method_name, run_name) + "/NER_BERT_BILSTM_CRF_KB_0.pt"
    if os.path.exists(model_exact_path):
        pass
    else:
        stra = Strategy(args, tagging_model, labeled_data, unlabeled_data,label_map, classifier, labelToIndex=labelToIndex)
        valid_acc_prev, valid_f1_prev = stra.train( -1, run_name, len(labeled_dataset), train_dataloader, dev_dataloader, test_dataloader, 0, 0, 0, model_path=model_path)
        print('time cost: ', (time.time()-start)/3600) 
    # log.flush()


    # Added
    classifier_model_path = os.path.join('checkpoint', 'classifier')
    classifier_model_exact_path = os.path.join('checkpoint/classifier') + "/NER_BERT_BILSTM_CRF_KB_CLASSIFIER_0.pt"
    if os.path.exists(classifier_model_exact_path):
        pass
    else:
        print("Original classifier model is not present. Train the new model")
        classifier_stra = Strategy(args, tagging_model, labeled_data, unlabeled_data,label_map, classifier, labelToIndex=labelToIndex)
        classifier_valid_acc_prev, classifier_valid_f1_prev = classifier_stra.train_classifier(-1, len(all_train_examples), classifier_train_dataloader, dev_dataloader, 0, 0, 0, model_path=classifier_model_path, number_of_epochs=50)
        print('time cost: ', (time.time()-start)/3600) 
    # log.flush()
    


    ####################################################
    # load tagging model best ckpt
    print('run_name_init', args.model_name)
    model_exact_path = os.path.join('checkpoint/'+ method_name, run_name) + "/NER_BERT_BILSTM_CRF_KB_0.pt"
    if os.path.exists(model_exact_path):
        checkpoint = torch.load(model_exact_path, map_location='cpu')
        start_epoch = checkpoint['epoch'] + 1
        valid_acc_prev = checkpoint['valid_acc']
        valid_f1_prev = checkpoint['valid_f1']
        pretrained_dict = checkpoint['model_state']
        net_state_dict = tagging_model.state_dict()
        pretrained_dict_selected = {k: v for k, v in pretrained_dict.items() if k in net_state_dict}
        net_state_dict.update(pretrained_dict_selected)
        tagging_model.load_state_dict(net_state_dict)
        print('Loaded the pretrain NER_BERT_BILSTM_CRF_KB model, epoch:', checkpoint['epoch'], 'valid acc:',
              checkpoint['valid_acc'], 'valid f1:', checkpoint['valid_f1'])
    else:
        print("init model not loaded!")
        raise NotImplementedError()

    classifier_model_exact_path = os.path.join('checkpoint/classifier') + "/NER_BERT_BILSTM_CRF_KB_CLASSIFIER_0.pt"
    if os.path.exists(classifier_model_exact_path):
        classifier_checkpoint = torch.load(classifier_model_exact_path, map_location='cpu')
        classifier_start_epoch = classifier_checkpoint['epoch'] + 1
        classifier_valid_acc_prev = classifier_checkpoint['valid_acc']
        classifier_valid_f1_prev = classifier_checkpoint['valid_f1']
        classifier_pretrained_dict = classifier_checkpoint['model_state']
        classifier_net_state_dict = classifier.state_dict()
        classifier_pretrained_dict_selected = {k: v for k, v in classifier_pretrained_dict.items() if k in classifier_net_state_dict}
        classifier_net_state_dict.update(classifier_pretrained_dict_selected)
        classifier.load_state_dict(classifier_net_state_dict)
        print('Loaded the pretrain NER_BERT_BILSTM_CRF_KB_CLASSIFIER model, epoch:', classifier_checkpoint['epoch'], 'valid acc:',
              classifier_checkpoint['valid_acc'], 'valid f1:', classifier_checkpoint['valid_f1'])
        
    else:
        print("init calssifier model not loaded")
        raise NotImplementedError()


    #2. coarse sampling
    run_name = "select"
    samp_select_all=[]
    select_size = int(len(all_train_examples) * args.data_ratio)
    method_name = args.coarse_sampling
    model_path = os.path.join('checkpoint', method_name)
    if not os.path.exists(model_path):
        os.makedirs(model_path)
 
    # 2.1. selecting samples by the AL strategy
    if method_name == 'entropy':
        sampler = EntropySampling(args, tagging_model, labeled_data, unlabeled_data, label_map, classifier, labelToIndex=labelToIndex)
    if method_name == 'margin':
        sampler = MarginSampling(args, tagging_model, labeled_data, unlabeled_data, label_map, classifier, labelToIndex=labelToIndex)
    if method_name == 'bald':
        sampler = BALDDropout(args, tagging_model, labeled_data, unlabeled_data, label_map, classifier, labelToIndex=labelToIndex)
    if method_name == 'bald2':
        sampler = BALDDropout2(args, tagging_model, labeled_data, unlabeled_data, label_map, classifier, labelToIndex=labelToIndex)
    if method_name == 'random':
        sampler = RandomSampling(args, tagging_model, labeled_data, unlabeled_data, label_map, classifier, labelToIndex=labelToIndex)
    print('----------------coarse samp_select------------ ')
    # log.flush()

    # 2.2 start selection interation
    for ind_iter in range(args.n_iter):
        # Added
        if ind_iter % 5 == 0:
            sampler.train_classifier(ind_iter, len(all_train_examples), classifier_train_dataloader, classifier_dev_dataloader, 0, 0, 0, model_path=classifier_model_path, number_of_epochs=2)

        # Added for classifier
        classifier_labeled_examples = conllProcessor._create_examples(sampler.labeled_data)
        classifier_labeled_dataset = NerDataset(classifier_labeled_examples, tokenizer, label_map, max_seq_length)
        classifier_labeled_dataloader = data.DataLoader(dataset=classifier_labeled_dataset,
                                            batch_size=args.classifier_batch_size,
                                            shuffle=False, # No shuffling, sequential access
                                            num_workers=4,
                                            collate_fn=NerDataset.pad)

        # Added for classifier
        classifier_unlabeled_examples = conllProcessor._create_examples(sampler.unlabeled_data)
        classifier_unlabeled_dataset = NerDataset(classifier_unlabeled_examples, tokenizer, label_map, max_seq_length)
        classifier_unlabeled_dataloader = data.DataLoader(dataset=classifier_unlabeled_dataset,
                                            batch_size=args.classifier_batch_size,
                                            shuffle=False, # No shuffling, sequential access
                                            num_workers=4,
                                            collate_fn=NerDataset.pad)


        print('--------------------ind_iter: ', ind_iter)
        samp_select = sampler.query(select_size, classifier_labeled_dataloader, classifier_unlabeled_dataloader)
        # assert len(samp_select) > 0.5 * select_size
        samp_select_final = samp_select
        print('coarse sampling filter size now: ', len(samp_select))

        save_samp_select = False
        if save_samp_select:
            save_samp_select_path = './samp_select/' + run_name
            if not os.path.exists(save_samp_select_path):
                os.makedirs(save_samp_select_path)
            samp_select_all.extend(samp_select_final)
            with open(save_samp_select_path + "/" + method_name + '.pkl','wb') as f:
                pickle.dump(samp_select_all,f)    
        print('-----------------updating dataset------- ')
        # update
        sampler.update_dataset(samp_select_final)
        print('ranking sampling filter size now: ', len(samp_select_final))
        # log.flush()

        labeled_examples = conllProcessor._create_examples( sampler.labeled_data )
        #unlabeled_examples = conllProcessor._create_examples( sampler.unlabeled_data )

        current_train_dataset = NerDataset(labeled_examples, tokenizer, label_map, max_seq_length)
        train_dataloader = data.DataLoader(dataset=current_train_dataset,
                                           batch_size=args.batch_size,
                                           shuffle=True,
                                           num_workers=4,
                                           collate_fn=NerDataset.pad)
        print('size of labeled_data: ', len(sampler.labeled_data ))  #
        print('size of unlabeled_data: ', len(sampler.unlabeled_data)) #
        print('-----------------retraining-----------------')
        valid_acc_prev, valid_f1_prev = sampler.train(ind_iter, run_name, len(labeled_examples), train_dataloader, dev_dataloader, test_dataloader, valid_acc_prev, valid_f1_prev, start_epoch, model_path)
        # log.flush()

    end = time.time()
    print('time cost: ', str(end-start))
    # log.flush()
    # log.close()