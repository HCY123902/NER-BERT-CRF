'''
Functions and Classes for read and organize data set
'''
import os
import torch
from torch.utils import data
import numpy as np
import random

import json

max_seq_length = 256

class InputExample(object):
    """A single training/test example for NER."""

    def __init__(self, guid, words, onto_labels, db_labels, labels, turn_label):
        """Constructs a InputExample.

        Args:
          guid: Unique id for the example(a sentence or a pair of sentences).
          words: list of words of sentence
          labels_a/labels_b: (Optional) string. The label seqence of the text_a/text_b. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        # list of words of the sentence,example: [EU, rejects, German, call, to, boycott, British, lamb .]
        self.words = words
        # list of onto label sequence of the sentence,like: [0,1,0,1,0,O]
        self.onto_labels = onto_labels
        # list of db label sequence of the sentence,like: [0,1,0,1,1,0]
        self.db_labels = db_labels
        # list of label sequence of the sentence,like: [B-ORG, O, B-MISC, O, O, O, B-MISC, O, O]
        self.labels = labels
        # Added for turn label
        self.turn_label = turn_label


class InputFeatures(object):
    """A single set of features of data.
    result of convert_examples_to_features(InputExample)
    """

    def __init__(self, guid, tokens, input_ids, onto_labels, db_labels, input_mask, segment_ids, predict_mask,
                 label_ids):
        self.guid = guid,
        self.tokens = tokens,
        self.input_ids = input_ids
        self.onto_labels = onto_labels
        self.db_labels = db_labels
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.predict_mask = predict_mask
        self.label_ids = label_ids


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_data(cls, input_file, labelToIndex, ratio=1):
        """
        Reads a BIO data.
        """
        # with open(input_file) as f:
        #     # out_lines = []
        #     out_lists = []
        #     entries = f.read().strip().split("\n\n")
        #     for entry in entries:
        #         words = []
        #         labels = []
        #         onto_labels = []
        #         db_labels = []
        #         for line in entry.splitlines():
        #             pieces = line.strip().split()
        #             if len(pieces) < 1:
        #                 continue
        #             if pieces[0] == "====":
        #                 words.append('[SEP]')
        #                 onto_labels.append(0)
        #                 db_labels.append(0)
        #                 labels.append('[SEP]')
        #                 continue

        #             word = pieces[0]
        #             # if word == "-DOCSTART-" or word == '':
        #             #     continue
        #             words.append(word)
        #             onto_labels.append(int(pieces[1]))
        #             db_labels.append(int(pieces[2]))
        #             labels.append(pieces[-1])

        #         out_lists.append([words, onto_labels, db_labels, labels])

        # if ratio<1:
        #     labeled_data = random.sample(out_lists, int(ratio*len(out_lists)))
        #     unlabeled_data = [item for item in out_lists if item not in labeled_data]
        # else:
        #     labeled_data = out_lists
        #     unlabeled_data = None

        # return labeled_data, unlabeled_data

        with open(input_file) as f:
            # out_lines = []
            out_lists = []
            entries = f.read().strip().split("\n\n")
            for entry in entries:
                words = []
                labels = []
                onto_labels = []
                db_labels = []
                turn_label = None
                for line in entry.splitlines():
                    pieces = line.strip().split()
                    if len(pieces) < 1:
                        continue
                    if pieces[0] == "====":
                        words.append('[SEP]')
                        onto_labels.append(0)
                        db_labels.append(0)
                        labels.append('[SEP]')
                        continue
                    # Added for turn labels
                    elif len(pieces) == 1:
                        assert pieces[0] in labelToIndex
                        turn_label = labelToIndex[pieces[0]]
                        continue

                    word = pieces[0]
                    # if word == "-DOCSTART-" or word == '':
                    #     continue
                    words.append(word)
                    onto_labels.append(int(pieces[1]))
                    db_labels.append(int(pieces[2]))
                    labels.append(pieces[-1])

                assert turn_label is not None
                out_lists.append([words, onto_labels, db_labels, labels, turn_label])
        if ratio<1:
            labeled_data = random.sample(out_lists, int(ratio*len(out_lists)))
            unlabeled_data = [item for item in out_lists if item not in labeled_data]
        else:
            labeled_data = out_lists
            unlabeled_data = None

        return labeled_data, unlabeled_data


class CoNLLDataProcessor(DataProcessor):
    '''
    CoNLL-2003
    '''

    def __init__(self, labelToIndex):
        # self._label_types = [ 'X', '[CLS]', '[SEP]', 'O', 'I-LOC', 'B-PER', 'I-PER', 'I-ORG', 'I-MISC', 'B-MISC', 'B-LOC', 'B-ORG']
        self._label_types = ['X', '[CLS]', '[SEP]', 'O', 'I', 'B']
        self._num_labels = len(self._label_types)
        self._label_map = {label: i for i,
                                        label in enumerate(self._label_types)}

        self.labelToIndex = labelToIndex

    def get_train_data_list_ratio(self, data_dir, data_ratio):
        labeled_data, unlabeled_data = self._read_data(os.path.join(data_dir, "train.txt"), self.labelToIndex, ratio=data_ratio)
        return labeled_data, unlabeled_data

    def get_train_examples(self, data_dir):
        labeled_data, _ = self._read_data(os.path.join(data_dir, "train.txt"), self.labelToIndex)
        return self._create_examples(labeled_data)

    def get_dev_examples(self, data_dir):
        labeled_data, _ = self._read_data(os.path.join(data_dir, "valid.txt"), self.labelToIndex)
        return self._create_examples(labeled_data)

    def get_test_examples(self, data_dir):
        labeled_data, _ = self._read_data(os.path.join(data_dir, "test.txt"), self.labelToIndex)
        return self._create_examples(labeled_data)

    def get_labels(self):
        return self._label_types

    def get_num_labels(self):
        return self.get_num_labels

    def get_label_map(self):
        return self._label_map

    def get_start_label_id(self):
        return self._label_map['[CLS]']

    def get_stop_label_id(self):
        return self._label_map['[SEP]']

    def _create_examples(self, all_lists):
        # examples = []
        # for (i, one_lists) in enumerate(all_lists):
        #     guid = i
        #     words = one_lists[0]
        #     onto_labels = one_lists[1]
        #     db_labels = one_lists[2]
        #     labels = one_lists[-1]
        #     examples.append(InputExample(
        #         guid=guid, words=words, onto_labels=onto_labels, db_labels=db_labels, labels=labels))
        # return examples
        examples = []
        for (i, one_lists) in enumerate(all_lists):
            guid = i
            words = one_lists[0]
            onto_labels = one_lists[1]
            db_labels = one_lists[2]
            # Adjusted
            labels = one_lists[-2]
            turn_label = one_lists[-1]
            examples.append(InputExample(
                guid=guid, words=words, onto_labels=onto_labels, db_labels=db_labels, labels=labels, turn_label=turn_label))
        return examples

    def _create_examples2(self, lines):
        examples = []
        for (i, line) in enumerate(lines):
            guid = i
            text = line[0]
            ner_label = line[-1]
            examples.append(InputExample(
                guid=guid, text_a=text, labels_a=ner_label))
        return examples


def example2feature(example, tokenizer, label_map, max_seq_length):
    add_label = 'X'
    # tokenize_count = []
    tokens = ['[CLS]']
    onto_labels = [0]
    db_labels = [0]
    predict_mask = [0]
    label_ids = [label_map['[CLS]']]
    for i, w in enumerate(example.words):
        # use bertTokenizer to split words
        # 1996-08-22 => 1996 - 08 - 22
        # sheepmeat => sheep ##me ##at
        if w == '[SEP]':
            sub_words = ['[SEP]']
        else:
            sub_words = tokenizer.tokenize(w)

        if not sub_words:
            sub_words = ['[UNK]']
        # tokenize_count.append(len(sub_words))

        tokens.extend(sub_words)
        for j in range(len(sub_words)):
            if j == 0:
                if sub_words[0] == '[SEP]':
                    predict_mask.append(0)
                else:
                    predict_mask.append(1)
                label_ids.append(label_map[example.labels[i]])
            else:
                # '##xxx' -> 'X' (see bert paper)
                predict_mask.append(0)
                label_ids.append(label_map[add_label])

            onto_labels.append(example.onto_labels[i])
            db_labels.append(example.db_labels[i])

    # truncate
    if len(tokens) > max_seq_length - 1:
        print('Example No.{} is too long, length is {}, truncated to {}!'.format(example.guid, len(tokens),
                                                                                 max_seq_length))
        tokens = tokens[0:(max_seq_length - 1)]
        onto_labels = onto_labels[0:(max_seq_length - 1)]
        db_labels = db_labels[0:(max_seq_length - 1)]
        predict_mask = predict_mask[0:(max_seq_length - 1)]
        label_ids = label_ids[0:(max_seq_length - 1)]

    first_sep = tokens.index('[SEP]')

    tokens.append('[SEP]')
    onto_labels.append(0)
    db_labels.append(0)
    predict_mask.append(0)
    label_ids.append(label_map['[SEP]'])

    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    segment_ids = [0] * (first_sep + 1) + [1] * (len(input_ids) - first_sep - 1)
    input_mask = [1] * len(input_ids)

    feat = InputFeatures(
        guid=example.guid,
        tokens=tokens,
        input_ids=input_ids,
        onto_labels=onto_labels,
        db_labels=db_labels,
        input_mask=input_mask,
        segment_ids=segment_ids,
        predict_mask=predict_mask,
        label_ids=label_ids,
        turn_label=example.turn_label)

    return feat


class NerDataset(data.Dataset):
    def __init__(self, examples, tokenizer, label_map, max_seq_length, mode = 'train'):
        self.examples = examples
        self.tokenizer = tokenizer
        self.label_map = label_map
        self.max_seq_length = max_seq_length
        self.mode = mode

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        feat = example2feature(self.examples[idx], self.tokenizer, self.label_map, self.max_seq_length)

        # Adjusted
        # return feat.input_ids, feat.onto_labels, feat.db_labels, feat.input_mask, feat.segment_ids, feat.predict_mask, feat.label_ids
        return feat.input_ids, feat.onto_labels, feat.db_labels, feat.input_mask, feat.segment_ids, feat.predict_mask, feat.label_ids, feat.turn_label
    
    @classmethod
    def pad(cls, batch):
        seqlen_list = [len(sample[0]) for sample in batch]
        maxlen = np.array(seqlen_list).max()

        f = lambda x, seqlen: [sample[x] + [0] * (seqlen - len(sample[x])) for sample in batch]  # 0: X for padding
        input_ids_list = torch.LongTensor(f(0, maxlen))
        onto_labels_list = torch.LongTensor(f(1, maxlen))
        db_labels_list = torch.LongTensor(f(2, maxlen))
        input_mask_list = torch.LongTensor(f(3, maxlen))
        segment_ids_list = torch.LongTensor(f(4, maxlen))
        predict_mask_list = torch.ByteTensor(f(5, maxlen))
        label_ids_list = torch.LongTensor(f(6, maxlen))

        # Added for turn label
        turn_labels = [sample[7] for sample in batch]
        turn_labels = torch.LongTensor(turn_labels)

        # return input_ids_list, onto_labels_list, db_labels_list, input_mask_list, segment_ids_list, predict_mask_list, label_ids_list
        return input_ids_list, onto_labels_list, db_labels_list, input_mask_list, segment_ids_list, predict_mask_list, label_ids_list, turn_labels
        
    
    @classmethod
    def padselect(cls, batch):
        seqlen_list = [len(sample[0]) for sample in batch]
        maxlen = max_seq_length
        #maxlen = np.array(seqlen_list).max()

        f = lambda x, seqlen: [sample[x] + [0] * (seqlen - len(sample[x])) for sample in batch]  # 0: X for padding
        input_ids_list = torch.LongTensor(f(0, maxlen))
        onto_labels_list = torch.LongTensor(f(1, maxlen))
        db_labels_list = torch.LongTensor(f(2, maxlen))
        input_mask_list = torch.LongTensor(f(3, maxlen))
        segment_ids_list = torch.LongTensor(f(4, maxlen))
        predict_mask_list = torch.ByteTensor(f(5, maxlen))
        label_ids_list = torch.LongTensor(f(6, maxlen))

        # Added for turn label
        turn_labels = [sample[7] for sample in batch]
        turn_labels = torch.LongTensor(turn_labels)

        # return input_ids_list, onto_labels_list, db_labels_list, input_mask_list, segment_ids_list, predict_mask_list, label_ids_list
        return input_ids_list, onto_labels_list, db_labels_list, input_mask_list, segment_ids_list, predict_mask_list, label_ids_list, turn_labels

