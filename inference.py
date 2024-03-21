# %reload_ext autoreload
# %autoreload 2

import argparse
import sys
import os
import json
import pickle as pkl
import collections
import numpy as np
import pandas as pd
import sklearn.metrics
import tensorflow as tf
from modeling import get_model_cls
from analysis_util import read_file, load_all, get_best_trial
from encode_data import Encoder, Mapping


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_dir', type=str,
                        # default='recommender/v3/outputs/nn_outputs/best_model',
                        help=('the model dir that stores the best model'))

    parser.add_argument('--data_dir', type=str,
                        # default='recommender/v3/data/data',
                        help=('the data directory stores the test data.'))

    parser.add_argument('--data_file', type=str,
                        # default='test.csv',
                        help=('the data file that we want to run inference'))

    parser.add_argument('--metadata_dir', type=str,
                        # default='recommender/v3/config',
                        help=('the config directory.'))

    parser.add_argument('--metadata_file', type=str,
                        # default='metadata.json',
                        help=('what is the corresponding metadata file?'))

    parser.add_argument('--encoder_dir', type=str,
                        # default='recommender/v3/data/encoded_data',
                        help=('the encoded data directory.'))

    parser.add_argument('--encoder_file', type=str,
                        # default='encoder.pkl',
                        help=('what is the encoder file?'))

    parser.add_argument('--output_dir', type=str,
                        # default='path/to/save/outputs',
                        help=('directory to save the inference results.'))

    args = parser.parse_args()

    path_to_data = os.path.join(args.data_dir, args.data_file)
    path_to_encoder = os.path.join(args.encoder_dir, args.encoder_file)
    path_to_metadata = os.path.join(args.metadata_dir, args.metadata_file)

    ## load inference data
    df = read_file(path_to_data)

    ## load the encoder
    encoder_file = open(path_to_encoder, 'rb')
    encoder = pkl.load(encoder_file)
    y, X, _ = encoder.transform(df)

    ## load best model from all trials
    best_trial = get_best_trial(args.model_dir)

    ## load the model config of the best model
    model_config_file = os.path.join(best_trial, 'model_config.json')
    with open(model_config_file, 'r') as f:
        model_config = json.load(f)

    model = get_model_cls(model_config['model_type'])(encoder.text_config, model_config)
    model.load(best_trial)
    print(model.model.summary())

    pred = model.predict(X, output_dir=args.output_dir)

    if model_config['task_type'] == 'regression':
        mse = sklearn.metrics.mean_squared_error(y, pred)
        print('mean square error is {}'.format(mse))
    elif model_config['task_type'] == 'classification':
        acc = sklearn.metrics.accuracy_score(y, pred)
        precision = sklearn.metrics.precision_score(y, pred)
        recall = sklearn.metrics.recall_score(y, pred)
        print('accuracy is {}, precision is {}, and recall is {}'.format(acc, precision, recall))
    else:
        raise ValueError('task type is not recognized!')

    ## save prediction results. it is a table contains TenantId, y, and pred.
    with open(path_to_metadata, 'r') as f:
        metadata = json.load(f)
    output_col = metadata['output_label']
    pred_col = []
    for c in output_col:
        w = c + '_pred'
        pred_col.append(w)
    print(pred_col)

    results_df = pd.DataFrame(data=pred, index=df.TenantId, columns=pred_col)
    results_df = results_df.reset_index()
    for c in output_col:
        results_df[c] = df[c]

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    path_to_save = os.path.join(args.output_dir, 'results.csv')

    results_df.to_csv(path_to_save, index=None)


if __name__ == '__main__':
    main()









