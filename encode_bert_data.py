#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import time
import collections
import argparse
import os
import json
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
import pickle


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument('--text_col', type=str,
        # default='metadata.json',
        help=('which column stores the text data?'))

    parser.add_argument('--label_col', type=str,
        # default='metadata.json',
        help=('which column stores the labels?'))

    parser.add_argument('--train_file', type=str,
        # default='metadata.json',
        help=('which train file will be used?'))

    parser.add_argument('--dev_file', type=str,
        # default='metadata.json',
        help=('which dev file will be used?'))

    parser.add_argument('--test_file', type=str,
        # default='metadata.json',
        help=('which test file will be used?'))

    parser.add_argument('--output_dir', type=str,
        # default='/data/home/t-chepan/projects/MS-intern-project/raw_data',
        help=('directory to save the encoded data.'))


    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    label_encoder = LabelEncoder()

    for i, (file, name) in enumerate(zip([args.train_file, args.dev_file, args.test_file], ['train.tsv', 'dev.tsv', 'test.tsv'])):
        df = read_file(file)
        print("Processing {} data: {} examples in total".format(name, df.shape[0]))
        texts = extract_text(df, args.text_col)
        if i == 0:
            labels = list(label_encoder.fit_transform(df[args.label_col]))
        else:
            labels = list(label_encoder.transform(df[args.label_col]))

        with open(os.path.join(args.output_dir, name), 'w') as f:
            f.write('label\ttext\n')
            for text, label in zip(texts, labels):
                f.write('{}\t{}\n'.format(label, text))

    print('Saving label encoder...')
    with open(os.path.join(args.output_dir, 'label_encoder.pkl'), 'wb') as f:
        pickle.dump(label_encoder, f)
    print('Processing done!')

    
def extract_text(df, text_col):
    texts = list(df[text_col])
    cleaned_texts = []
    for t in texts:
        cleaned_texts.append(clean_text(t))
    return cleaned_texts


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


def clean_text(text):
    s = text
    s = str(s).replace('\n', ' ')
    s = s.replace('\t', ' ')
    s = s.replace('\r', ' ')
    return s


if __name__ == '__main__':
    main()




