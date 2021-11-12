import numpy as np
import os
import torch
import random
from sklearn.metrics import f1_score


def seed_torch(seed=44):
    torch.manual_seed(seed)
    cuda_yes = torch.cuda.is_available()
    if cuda_yes:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False

def defined_f1_score(y_true, y_pred):
    '''
    0,1,2,3 are [CLS],[SEP],[X],O
    '''
    ignore_id = 3

    num_proposed = len(y_pred[y_pred > ignore_id])
    num_correct = (np.logical_and(y_true == y_pred, y_true > ignore_id)).sum()
    num_gold = len(y_true[y_true > ignore_id])

    try:
        precision = num_correct / num_proposed
    except ZeroDivisionError:
        precision = 1.0

    try:
        recall = num_correct / num_gold
    except ZeroDivisionError:
        recall = 1.0

    try:
        f1 = 2 * precision * recall / (precision + recall)
    except ZeroDivisionError:
        if precision * recall == 0:
            f1 = 1.0
        else:
            f1 = 0

    return precision, recall, f1


def spanf1_score(y_true, y_pred):
    def get_spans(label_list):
        spans = []
        s = ""
        start = 0
        for i, l in enumerate(label_list):
            if i == len(label_list) - 1:
                if l <= 3:  # end with O
                    if len(s) > 0:
                        spans.append((s, (start, len(label_list) - 2)))
                else:  # not end with O
                    s += str(l)
                    spans.append((s, (start, i)))

            else:  # not last token
                if l <= 3:  # outside
                    if len(s) == 0:
                        continue
                    else:
                        spans.append((s, (start, i - 1)))
                        s = ""
                else:
                    if len(s) == 0:
                        start = i
                        s += str(l)
                    else:
                        s += str(l)
        return spans

    true_spans = get_spans(y_true)
    pred_spans = get_spans(y_pred)

    correct = 0
    for span in pred_spans:
        if span in true_spans:
            correct += 1

    if len(pred_spans) == 0:
        p = 1.0 * correct / 1
    else:
        p = 1.0 * correct / len(pred_spans)
    if len(true_spans) == 0:
        r = 1.0 * correct / 1
    else:
        r = 1.0 * correct / len(true_spans)

    if p + r == 0:
        p = 1

    f1 = 2.0 * p * r / (p + r)
    return p, r, f1

def warmup_linear(x, warmup=0.002):
    if x < warmup:
        return x / warmup
    return 1.0 - x
