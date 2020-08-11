import os
import json
import pandas as pd
import pickle as pkl
import numpy as np
from encode_data import Encoder, Mapping
from modeling import get_model_cls


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
