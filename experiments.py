"""
random search for hyperparameters:
1. user provide a search space
2. we generate a set of hyperparameters (as model_config) within this space
3. feed the medel_config to modeling part
4. save the model, result and the model_config
"""

import argparse
import os
import json
import shutil
import tensorflow as tf
import numpy as np
from encode_data import Mapping, Encoder
from modeling import Model, get_model_cls
import sys
from keras import backend as K

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#   try:
#     for gpu in gpus:
#       tf.config.experimental.set_memory_growth(gpu, True)
#   except RuntimeError as e:
#     print(e)

#
# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#   try:
#     tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=10024)])
#   except RuntimeError as e:
#     print(e)

def load_encoded_data(data_path):
    if os.path.exists(data_path):
        encoded_data = np.load(data_path, mmap_mode='r')
    else:
        encoded_data = None
    return encoded_data

def hyper_tune_non_bert_model(num_trials, search_space, default_model_config,
               y_train, X_train_struc, X_train_text,
               y_dev, X_dev_struc, X_dev_text, text_config):
    #######################################################################################
    ## For each trial, update default model_config based on search_space and train model ##
    #######################################################################################
    for i in range(num_trials):

        print('Running trial number {}!'.format(i))
        model_config = sample_modelconfig(search_space, default_model_config)
        model_name = 'model_{}'.format(i)
        print('*' * 50)

        model_config = Mapping(model_config)

        model_config.output_dir = os.path.join(default_model_config.output_dir, model_name)
        if not os.path.exists(model_config.output_dir):
            os.makedirs(model_config.output_dir)

        print('F' * 20)
        print('model_output_dir: ' + model_config['output_dir'])
        print('model_batch_size: {}'.format(model_config['batch_size']))
        print('F' * 20)

        model = get_model_cls(model_config.model_type)(text_config, model_config)
        experiment_output = model.train(y_train, X_train_struc, X_train_text, y_dev, X_dev_struc, X_dev_text)


        ## save output and model_config ##
        experiment_output_path = os.path.join(model_config.output_dir, 'output.json')
        with open(experiment_output_path, 'w') as f:
            json.dump(experiment_output, f, indent=4)

        model_config_savepath = os.path.join(model_config.output_dir, 'model_config.json')
        with open(model_config_savepath, 'w') as mf:
            json.dump(model_config, mf, indent=4)
        print('*' * 50)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--encoded_data_dir', type=str,
                        # default='/data/home/t-chepan/projects/MS-intern-project/data',
                        help=('the input data dir. should contain the .tsv files (or other data files)'))

    # this is optional 
    parser.add_argument('--data_name', type=str,
                        # default='KICK',
                        help=('which version of data will be used? (kickstarter Or indiegogo?)'))

    parser.add_argument('--search_space_dir', type=str,
                        # default='path/to/search_space.json',
                        help=('where to load the search space file?'))

    parser.add_argument('--search_space_filename', type=str,
                        # default='path/to/search_space.json',
                        help=('search space file name?'))

    parser.add_argument('--output_dir', type=str,
                        # default='path/to/save/outputs',
                        help=('directory to save the trained model and related model_config.'))

    parser.add_argument('--task_type', type=str,
                        default='classification',
                        help=('what is the type of this task? (classification or regression?)'))
    parser.add_argument('--metric', type=str,
                        default='acc',
                        help=('what metric will be used in this task? (acc, auc, or mse?)'))

    parser.add_argument('--num_classes', type=int,
                        # default='classification',
                        help=('what is the number of classes (classification)?'))

    parser.add_argument('--num_outputs', type=int,
                        default=1,
                        help=('what is the number of outputs (single or multi-outputs)?'))

    parser.add_argument('--model_type', type=str,
                        # default='mlp',
                        help=(
                            'what type of NN model you want to try? (mlp, bert, logistic_regression, random_forest, or svm?)'))

    parser.add_argument('--num_trials', type=int,
                        default=1,
                        help=('how many trials you want to run the model?'))

    ### BERT required parameters ###
    parser.add_argument('--bert_dir', type=str, default=None,
                        help=('The config json file corresponding to the pre-trained BERT model.'))

    args = parser.parse_args()

    if args.bert_dir is not None:
        if not os.path.exists(args.bert_dir):
            print('{} not found'.format(args.bert_dir))
            sys.exit()
        else:
            from bert import bert_classifier  ## have to download and save bert pre-trained nn_outputs in "bert" folder

    if args.data_name is not None and args.encoded_data_dir is not None:
        path_to_data = os.path.join(args.encoded_data_dir, args.data_name)
        path_to_save = os.path.join(args.output_dir, args.data_name)

    elif args.data_name is None and args.encoded_data_dir is not None:
        path_to_data = args.encoded_data_dir
        path_to_save = args.output_dir

    else:
        raise argparse.ArgumentTypeError("args.data_name or args.encoded_data_dir can't be recognized.")


    if not os.path.exists(path_to_save):
        os.makedirs(path_to_save)

    ###########################################
    ## sample model config from search space ##
    ###########################################

    if args.task_type is not None and args.num_outputs is not None:
        print('This is a {} task, and you are choosing {} as the model type and {} as the metric!'.format(args.task_type, args.model_type, args.metric))
        default_model_config = create_default_modelconfig(args.task_type, args.metric,
                                                          args.num_classes, args.num_outputs,
                                                          args.model_type, path_to_save)
        default_model_config['encoded_data_dir'] = args.encoded_data_dir
    else:
        raise ValueError('You are missing task_type or num_outputs or both!')

        ## load search space file which is provided by users ##
    path_to_search_file = os.path.join(args.search_space_dir, args.search_space_filename)

    with open(path_to_search_file, 'r') as f:
        search_space = json.load(f)
    search_space = Mapping(search_space)


    if args.model_type != 'bert':
        ###########################################
        ## load encoded training set and dev set ##
        ###########################################

        y_train_path = os.path.join(path_to_data, 'y_train.npy')
        y_train = load_encoded_data(y_train_path)
        if y_train is None:
            raise ValueError('y_train is not found!')

        X_train_struc_path = os.path.join(path_to_data, 'X_train_struc.npy')
        X_train_struc = load_encoded_data(X_train_struc_path)

        X_train_text_path = os.path.join(path_to_data, 'X_train_text.npy')
        X_train_text = load_encoded_data(X_train_text_path)

        y_dev_path = os.path.join(path_to_data, 'y_dev.npy')
        y_dev = load_encoded_data(y_dev_path)
        if y_dev is None:
            raise ValueError('y_dev is not found!')

        X_dev_struc_path = os.path.join(path_to_data, 'X_dev_struc.npy')
        X_dev_struc = load_encoded_data(X_dev_struc_path)

        X_dev_text_path = os.path.join(path_to_data, 'X_dev_text.npy')
        X_dev_text = load_encoded_data(X_dev_text_path)

        text_config_path = os.path.join(path_to_data, 'text_config.json')
        if os.path.exists(text_config_path):
            with open(text_config_path, 'r') as f:
                text_config = json.load(f)
            text_config = Mapping(text_config)
        else:
            text_config = None

        if text_config is not None and text_config.mode == 'glove':
            embedding_matrix_path = text_config.embedding_matrix_path
            if os.path.exists(embedding_matrix_path):
                embedding_matrix = np.load(embedding_matrix_path, mmap_mode='r')
                text_config.embedding_matrix = embedding_matrix
            else:
                raise ValueError('embedding_matrix is not found!')
        else:
            embedding_matrix = None

        print('Start hyperparameter tuning for {} modle!'.format(args.model_type))

        hyper_tune_non_bert_model(args.num_trials,search_space, default_model_config,
                                  y_train, X_train_struc, X_train_text,
                                  y_dev, X_dev_struc, X_dev_text, text_config)

    elif args.model_type == 'bert':
        print('Start hyperparameter tuning for BERT model!')
        for i in range(args.num_trials):
            print('Running trial number {}!'.format(i))
            model_config = sample_modelconfig(search_space, default_model_config)
            model_name = 'model_{}'.format(i)
            print('*' * 50)

            model_config = Mapping(model_config)

            model_config.output_dir = os.path.join(default_model_config.output_dir, model_name)
            if not os.path.exists(model_config.output_dir):
                os.makedirs(model_config.output_dir)

            print('model_config: ' + model_config['output_dir'])
            if not args.bert_dir:
                raise ValueError('You must provide bert_dir when using BERT nn_outputs.')
            acc = bert_classifier.run_bert_classifier(
                model_config.output_dir, args.encoded_data_dir, model_config.num_classes, args.bert_dir,
                model_config.learning_rate, model_config.warmup_proportion, model_config.n_epochs,
                model_config.batch_size, model_config.batch_size, model_config.batch_size,
                do_train=True, do_eval=True, do_predict=False,
                do_lower_case=model_config.do_lower_case, max_seq_length=128,
                save_checkpoints_steps=1000)
            experiment_output = {'val_metric': 1 - acc['eval_accuracy']}

            ## save output and model_config ##
            experiment_output_path = os.path.join(model_config.output_dir, 'output.json')
            with open(experiment_output_path, 'w') as f:
                json.dump(experiment_output, f, indent=4)

            model_config_savepath = os.path.join(model_config.output_dir, 'model_config.json')
            with open(model_config_savepath, 'w') as mf:
                json.dump(model_config, mf, indent=4)
            print('*' * 50)

    else:
        raise ValueError('Cannot recognize model type {}.'.format(args.model_type))

    trial_metrics = []
    for trial_dir in os.listdir(default_model_config.output_dir):
        if trial_dir == '.DS_Store':
            continue
        output_file = os.path.join(default_model_config.output_dir, trial_dir, 'output.json')
        with open(output_file, 'r') as f:
            output = json.load(f)
            metric = output['val_metric']
        trial_metrics.append((metric, trial_dir))
    for i, (metric, trial_dir) in enumerate(sorted(
        trial_metrics, key=lambda x: x[0])[:5]):
        print('{}: {} {}'.format(i, metric, trial_dir))

    print('=' * 50)
    print('{} trials have been evaluated, the experiment finished successfully!'.format(args.num_trials))


def sample_modelconfig(search_space, default_model_config):
    model_config = default_model_config.copy()
    for k, v in search_space.items():
        if v[0] == 'linear_int':
            model_config[k] = np.random.randint(v[1][0], v[1][1])

        if v[0] == 'linear_cont':
            model_config[k] = np.random.uniform(v[1][0], v[1][1])

        if v[0] == 'log_cont':
            model_config[k] = np.random.uniform(np.log(v[1][0]), np.log(v[1][1]))
            model_config[k] = np.exp(model_config[k])

        if v[0] == 'log_int':
            model_config[k] = np.random.uniform(np.log(v[1][0]), np.log(v[1][1]))
            model_config[k] = int(np.round(np.exp(model_config[k])))

        if v[0] == 'category':
            model_config[k] = v[1][np.random.randint(len(v[1]))]

    return model_config


def create_default_modelconfig(task_type, metric, num_classes, num_outputs, model_type, output_dir):
    model_config = Mapping()
    model_config.model_type = model_type  ## default is 'mlp'.
    model_config.output_dir = output_dir
    model_config.task_type = task_type  ## 'classification' or 'regression'
    model_config.metric = metric
    if model_config.task_type == 'regression' and model_config.metric != 'mse':
        raise ValueError('model type is {} but evaluation metric is {}'.format(model_type,metric))
    model_config.num_classes = num_classes  ## number of classes for classification task
    model_config.num_outputs = num_outputs  ## ## number of outputs
    if model_type == 'mlp':
        model_config.combine = 'concate'  ## or 'attention'
        model_config.n_layers_dense = 2
        model_config.hidden_size_dense = 16
        model_config.n_layers_lstm = 2
        model_config.hidden_size_lstm = 32
        model_config.dropout_rate_lstm = 0.0
        model_config.n_layers_output = 2
        model_config.hidden_size_output = 32
        model_config.optimizer = 'adam'
        model_config.learning_rate = 0.001
        model_config.clipnorm = 5.0
        model_config.patience = 20
        model_config.n_epochs = 20
        model_config.batch_size = 1
        model_config.verbose = 0
    elif model_type == 'random_forest':
        model_config.n_trees = 10
    elif model_type == 'logistic_regression' or 'linear_regression':
        model_config.C = 0.01
    elif model_type == 'svm':
        model_config.C = 0.01
    elif model_type == 'bert':
        model_config.learning_rate = 5e-5
        model_config.warmup_proportion = 0.1
        model_config.n_epochs = 3.0
        model_config.batch_size = 32
        model_config.do_lower_case = True
    else:
        raise ValueError('Unknown model type: {}'.format(model_type))
    return model_config


if __name__ == '__main__':
    main()
