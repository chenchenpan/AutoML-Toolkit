
import argparse
import sys
import os
import json
import pickle as pkl
import collections
import numpy as np
import math
import pandas as pd
import sklearn.metrics
import tensorflow as tf
from modeling import get_model_cls
from analysis_util import read_file, count_df_rows_with_chunks, load_all, get_best_trial, calculate_eval_metric
from encode_data import Encoder, Mapping


def make_prediction_df(predictions, orig_df, index, pred_col_list, output_col_list):
    ## The results will be stored in a table contains TenantId, y, and pred.
    results_df = pd.DataFrame(data=predictions, index=index, columns=pred_col_list)
    results_df = results_df.reset_index()
    for c in output_col_list:
        results_df[c] = orig_df[c]
    return results_df


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_dir', type=str,
                        # default='recommender/v3/outputs/results',
                        help=('the dir that stores the model'))

    parser.add_argument('--model_name', type=str,
                        # default='best_model',
                        help=('which model we want to use for inference'))

    parser.add_argument('--data_dir', type=str,
                        # default='recommender/v3/data/data',
                        help=('the data directory stores the test data.'))

    parser.add_argument('--data_file', type=str,
                        # default='test.csv',
                        help=('the data file that we want to run inference'))

    parser.add_argument('--num_chunks', type=int,
                        help=('how many chunks we need to load and process the inference data'))

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

    parser.add_argument('--output_filename', type=str,
                        default='results.csv',
                        help=('Save the inference results with a specified filename.'))

    parser.add_argument('--output_dir', type=str,
                        # default='path/to/save/outputs',
                        help=('directory to save the inference results.'))

    args = parser.parse_args()

    path_to_model = os.path.join(args.model_dir, args.model_name)
    path_to_data = os.path.join(args.data_dir, args.data_file)
    path_to_encoder = os.path.join(args.encoder_dir, args.encoder_file)
    path_to_metadata = os.path.join(args.metadata_dir, args.metadata_file)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    path_to_save = os.path.join(args.output_dir, args.output_filename)

    ## load the encoder
    encoder_file = open(path_to_encoder, 'rb')
    encoder = pkl.load(encoder_file)

    ## load the model config of the best model
    if os.path.exists(path_to_model):
        best_trial = path_to_model
    else:
        raise ValueError('best model directory is not found!')

    model_config_file = os.path.join(best_trial, 'model_config.json')
    with open(model_config_file, 'r') as f:
        model_config = json.load(f)

    model = get_model_cls(model_config['model_type'])(encoder.text_config, model_config)
    model.load(best_trial)
    print(model.model.summary())


    ## load test data and save inference results
    ### The columns we need to store in results table: pred_col, output_col, and corresponding TenantId
    with open(path_to_metadata, 'r') as f:
        metadata = json.load(f)
    output_col = metadata['output_label']
    pred_col = []
    for c in output_col:
        w = c + '_pred'
        pred_col.append(w)
    print(pred_col)

    if args.num_chunks:
        print("Start to inference with chunks...")
        num_rows = count_df_rows_with_chunks(path_to_data,sep=',', chunksize=100000)

        accurate_size = math.ceil(num_rows / args.num_chunks)
        chunk_reader = pd.read_csv(path_to_data, header=0, error_bad_lines=True, iterator=True,
                                   chunksize=accurate_size)
        results_list = []
        for i, chunk in enumerate(chunk_reader):
            sub_df = pd.DataFrame(chunk)
            y, X, _ = encoder.transform(sub_df)
            # sub_output_dir = os.path.join(args.output_dir, 'predictions_'+str(i))
            # if not os.path.exists(sub_output_dir):
            #     os.makedirs(sub_output_dir)
            # pred = model.predict(X, output_dir=sub_output_dir)
            pred = model.predict(X, output_dir=args.output_dir)
            eval_metric = calculate_eval_metric(model_config['task_type'], y, pred)
            print('Chunk_{}th eval_metric is {}'.format(i, eval_metric))

            sub_results_df = make_prediction_df(pred, sub_df, sub_df.TenantId, pred_col, output_col)
            # sub_results_df = pd.DataFrame(data=pred, index=sub_df.TenantId, columns=pred_col)
            # sub_results_df = sub_results_df.reset_index()
            # for c in output_col:
            #     sub_results_df[c] = sub_df[c]
            results_list.append(sub_results_df)
        results_df = pd.concat(results_list, ignore_index=True)
        # Sanity check the number of test examples before and after inference
        if results_df.shape[0] != num_rows:
            raise ValueError('The shape of inference results does not match the original test data size!')

    else:
        df = read_file(path_to_data)
        y, X, _ = encoder.transform(df)
        print("Successfully encoder the entire test data!!")

        predicts = model.predict(X, output_dir=args.output_dir)
        eval_metric = calculate_eval_metric(model_config['task_type'], y, predicts)
        print('The eval_metric on this testset is {}'.format(eval_metric))

        results_df = make_prediction_df(predicts, df, df.TenantId, pred_col, output_col)


    # results_df = pd.DataFrame(data=predicts, index=df.TenantId, columns=pred_col)
    # results_df = results_df.reset_index()
    # for c in output_col:
    #     results_df[c] = df[c]

    results_df.to_csv(path_to_save, index=False)


if __name__ == '__main__':
    main()