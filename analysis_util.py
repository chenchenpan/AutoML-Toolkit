import os
import json
import pandas as pd
import pickle as pkl
import numpy as np
import shutil
import sklearn
from modeling import get_model_cls


def calculate_eval_metric(task_type, y, pred):
    if task_type == 'regression':
        mse = sklearn.metrics.mean_squared_error(y, pred)
        print('mean square error is {}'.format(mse))
        eval_metric = mse
    elif task_type == 'classification':
        acc = sklearn.metrics.accuracy_score(y, pred)
        # precision = sklearn.metrics.precision_score(y, pred)
        # recall = sklearn.metrics.recall_score(y, pred)
        # print('accuracy is {}, precision is {}, and recall is {}'.format(acc, precision, recall))
        print('accuracy is {}'.format(acc))
        eval_metric = acc
    else:
        raise ValueError('task type is not recognized!')
    return eval_metric


def count_df_rows_with_chunks(filename, sep=',', chunksize=1000):
    chunk_reader = pd.read_csv(filename, sep=sep, header=0, error_bad_lines=True, iterator=True,
                               chunksize=chunksize)
    num_rows = 0
    for chunk_number, chunk in enumerate(chunk_reader):
        chunk_df = pd.DataFrame(chunk)
        num_rows += chunk_df.shape[0]
    print("Total number of rows is {}".format(num_rows))
    return num_rows


def read_file(path):
    filename, file_extension = os.path.splitext(path)
    if file_extension == '.csv':
        sep = ','
    elif file_extension == '.tsv':
        sep = '\t'
    else:
        raise ValueError('Unknown type of file: {}. Please add .csv or .tsv.'.format(path))
    df = pd.read_csv(path, sep=sep)
    return df


# Find the best model.
def load_encoder(encoded_data_dir):
    encoder_file = os.path.join(encoded_data_dir, 'encoder.pkl')
    with open(encoder_file, 'rb') as f:
        encoder = pkl.load(f)
    if encoder.text_config is not None and encoder.text_config.mode == 'glove':
        embedding_file = os.path.join(encoded_data_dir, 'encoder.pkl')
        embedding_matrix = np.load(embedding_file)
        encoder.text_config.embedding_matrix = embedding_matrix
    return encoder


def load_model(trial_dir, encoder):
    model_config_file = os.path.join(trial_dir, 'model_config.json')
    with open(model_config_file, 'r') as f:
        model_config = json.load(f)
    model = get_model_cls(model_config['model_type'])(encoder.text_config, model_config)
    model.load(trial_dir)
    return model


def load_all(trial_dir):
    model_config_file = os.path.join(trial_dir, 'model_config.json')
    with open(model_config_file, 'r') as f:
        model_config = json.load(f)
    encoded_data_dir = model_config['encoded_data_dir']
    encoder = load_encoder(encoded_data_dir)
    model = get_model_cls(model_config['model_type'])(encoder.text_config, model_config)
    model.load(trial_dir)
    return model, encoder  


def get_best_trial(output_dir):
    best_trial = None
    best_metric = None
    for trial_name in os.listdir(output_dir):
        trial_dir = os.path.join(output_dir, trial_name)
        if trial_dir == '.DS_Store':
            continue
        with open(os.path.join(trial_dir, 'output.json'), 'r') as f:
            metric = json.load(f)['val_metric']
        if best_trial is None or best_metric > metric:
            best_metric = metric
            best_trial = trial_dir
    print('best metric: {}, best_trial: {}'.format(best_metric, best_trial))
    return best_trial

def copy_directory(old_dir, new_dir):
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)
    # return file list under the old directory
    file_list = os.listdir(old_dir)
    for file in file_list:
        # Add the file name to the current file path
        old_file_path = old_dir + '/' + file
        new_file_path = new_dir + '/' + file
        # If it is a file
        if os.path.isfile(old_file_path):
            print(old_file_path)
            print(new_dir)
            # copyfile The two functions must be files, not directories,
            shutil.copyfile(old_file_path, new_file_path)
        else: # If it is not a file, recurse the path of this folder
            copy_directory(old_file_path, new_file_path)