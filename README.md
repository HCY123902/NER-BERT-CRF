# NTD with diverse sampling
# Environment
1. python 3.6
1. pytorch 1.6.0
1. pytorch-pretrained-bert 0.4.0

# Usage
1. Train a seperate classifier model that classifies the goal of each user agent turn with `NER-BERT-CRF` and place the model at `./checkpoint/classifier/` with the name `NER_BERT_BILSTM_CRF_KB_CLASSIFIER_0.pt`
1. Place your dataset at `mmconv_data/`. The default NTD dataset has already been placed in the path.
1. Train and evaluate the model with `python -W ignore main_al.py --coarse_sampling {method name}`, where method name can be either `random`, `entropy`, `margin`, or `bald`
