import argparse

def parameter_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--classifier_batch_size',type= int, default = 128, help='batch size for classifier.')
    parser.add_argument('--classifier_model_path',type= int, default = 128, help='path of the classifier model.')

    parser.add_argument('--batch_size',type= int, default = 32, help='batch size for inference.')
    parser.add_argument('--data_ratio', type= float, default = 0.05, help='proportion of total training data used.')
    parser.add_argument('--init_data_ratio', type= float, default = 0.05, help='proportion of total training data used.')
    parser.add_argument('--pretrained_model',type= str, default = 'bert-base-cased', help='name of pretrained model as backbone.')
    parser.add_argument('--coarse_sampling', type=str, default='random',
                        help='name of coarse_sampling.')
    parser.add_argument('--n_iter', type= int, default = 19, help='selection iter.')
    parser.add_argument('--active_algo',type= str, default = 'entropy', help='active strategy method') 
    parser.add_argument('--turn_level',type=bool, default = True, help='whether work at turn level')
    parser.add_argument('--epoch',type= int, default = 10, help='training epochs')
    parser.add_argument('--finetune',action='store_true', help='--finetune is true')
    parser.add_argument("--lr", default=5e-6, type=float, required=False,
                        help="Learning rate.")
    parser.add_argument('--random_split',action='store_true', help='--random split the data')  


    # Input and output configs
    parser.add_argument("--model_name", default='bert_bilstm_crf_kb', type=str, required=False,
                        help="the base model")
    parser.add_argument("--task", default='classification', type=str, required=False,
                        help="the task to run bert ranker for")
    parser.add_argument("--output_dir", default='./active_output/', type=str, required=False,
                        help="the folder to output predictions")
    parser.add_argument("--save_model", default=False, type=bool, required=False,
                        help="Save trained model at the end of training.")

    #Training procedure
    parser.add_argument("--seed", default=44, type=str, required=False,
                        help="random seed")
    parser.add_argument("--num_epochs", default=10, type=int, required=False,
                        help="Number of epochs for training.")
    parser.add_argument("--num_training_instances", default=-1, type=int, required=False,
                        help="Number of training instances for training (if num_training_instances != -1 then num_epochs is ignored).")
    parser.add_argument("--max_gpu", default=4, type=int, required=False,
                        help="max gpu used")
    parser.add_argument("--num_validation_batches", default=-1, type=int, required=False,
                        help="Run validation for a sample of <num_validation_batches>. To run on all instances use -1.")
    parser.add_argument("--val_batch_size", default=64, type=int, required=False,
                        help="Validation and test batch size.")
    parser.add_argument("--sample_data", default=-1, type=int, required=False,
                         help="Amount of data to sample for training and eval. If no sampling required use -1.")

    #Model hyperparameters
    parser.add_argument("--transformer_model", default="bert-base-cased", type=str, required=False,
                        help="Bert model to use (default = bert-base-cased).")
    parser.add_argument("--max_seq_len", default=256, type=int, required=False,
                        help="Maximum sequence length for the inputs.")
    parser.add_argument("--loss_function", default="cross-entropy", type=str, required=False,
                        help="Loss function (default is 'cross-entropy').")
    parser.add_argument("--smoothing", default=0.1, type=float, required=False,
                        help="Smoothing hyperparameter used only if loss_function is label-smoothing-cross-entropy.")

    #Uncertainty estimation hyperparameters
    parser.add_argument("--predict_with_uncertainty_estimation", default=False, action="store_true", required=False,
                        help="Whether to use dropout at test time to get relevance (mean) and uncertainties (variance).")
    parser.add_argument("--num_forward_prediction_passes", default=10, type=int, required=False,
                        help="Number of foward passes with dropout to obtain mean and variance of predictions. "+
                             "Only used if predict_with_uncertainty_estimation == True.")

    return parser.parse_args()