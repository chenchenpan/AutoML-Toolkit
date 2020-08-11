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
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import np_utils
from pickle import dump


## check the package version
# print(pd.__version__)

# # Preprocess data steps:

# 1. split dataset
# 2. transfer datetime data
# 3. encode categorical data
# 4. encode boolean type data
# 5. normalize data

def main():

    parser = argparse.ArgumentParser()

    # # parameters for select input data and metedata configure files
    # parser.add_argument('--data_dir', type=str,
    #     # default='/data/home/t-chepan/projects/MS-intern-project/raw_data',
    #     help=('directory to load the raw data.'))

    # parser.add_argument('--data_name', type=str,
    #     # default='kickstarter',
    #     help=('which data will be used? (kickstarter Or indiegogo?)'))

    parser.add_argument('--metadata_file', type=str,
        # default='metadata.json',
        help=('which tabular metadata file will be used?'))

    parser.add_argument('--train_file', type=str,
        # default='metadata.json',
        help=('which train file will be used?'))

    parser.add_argument('--dev_file', type=str,
        # default='metadata.json',
        help=('which dev file will be used?'))

    parser.add_argument('--test_file', type=str,
        # default='metadata.json',
        help=('which test file will be used?'))

    # parameter for using text features
    parser.add_argument('--use_text_features', type=str2bool, nargs='?',
        const=True, default=False,
        help=('whether encode the text features or not?'))

    parser.add_argument('--encode_text_with', type=str,
        # default='tfidf',
        help=('how to encode the text features? (tfidf, glove)'))

    parser.add_argument('--glove_file', type=str,
        # default='/data/home/t-chepan/projects/MS-intern-project/raw_data',
        help=('directory to the GloVe file will be used. (e.g. glove.840B.300d.txt)'))

    parser.add_argument('--max_words', type=int,
        # default='/data/home/t-chepan/projects/MS-intern-project/raw_data',
        help=('what is the maximum number of words for encoding text?'))

    parser.add_argument('--max_sequence_length', type=int,
        # default='/data/home/t-chepan/projects/MS-intern-project/raw_data',
        help=('what is the maximum sequence length for encoding text?'))

    parser.add_argument('--output_dir', type=str,
        # default='/data/home/t-chepan/projects/MS-intern-project/raw_data',
        help=('directory to save the encoded data.'))


    args = parser.parse_args()

    ### load raw data and related metadata configure file
    # if args.data_name is not None and args.data_dir is not None:
    #     path_to_data = os.path.join(args.data_dir, args.data_name)
    #     path_to_save = os.path.join(args.output_dir, args.data_name)
    #     if not os.path.exists(path_to_save):
    #         os.makedirs(path_to_save)

    # elif args.data_name is None and args.data_dir is not None:
    #     path_to_data = args.data_dir
    #     path_to_save = args.output_dir

    # else:
    #     raise argparse.ArgumentTypeError(args.data_name + ' or ' + args.data_dir + " can't be recognized.")

    # if not os.path.exists(path_to_data):
    #     os.makedirs(path_to_data)

    # if not os.path.exists(path_to_save):
    #     os.makedirs(path_to_save)

    # train_path = os.path.join(path_to_data, args.train_file)
    # dev_path = os.path.join(path_to_data, args.dev_file)
    # test_path = os.path.join(path_to_data, args.test_file)


    print("Start to load data...")
    
    path_to_save = args.output_dir
    if not os.path.exists(path_to_save):
        os.makedirs(path_to_save)
        
    df_train = read_file(args.train_file)
    df_dev = read_file(args.dev_file)
    df_test = read_file(args.test_file)
    
    print('*' * 50)
    print('training set size is {}'.format(df_train.shape[0]))
    print('dev set size is {}'.format(df_dev.shape[0]))
    print('test set size is {}'.format(df_test.shape[0]))

    with open(args.metadata_file, 'r') as f:
        metadata = json.load(f)

    print("Processing data...")

    if args.use_text_features:
        mode = args.encode_text_with
        text_config = Mapping()
        text_config.mode = mode
        text_config.max_words = args.max_words          

        if mode == 'glove':
            # glove_file_path = os.path.join(args.glove_dir, args.glove_file)
            text_config.maxlen = args.max_sequence_length
            text_config.embeddings_index = open_glove(args.glove_file)
            text_config.embedding_dim = list(text_config.embeddings_index.values())[0].shape[-1]

        if mode != 'glove' and mode != 'tfidf':
            raise argparse.ArgumentTypeError(mode, "can't be recognized.")
    
    else:
        text_config = None
        
    encoder = Encoder(metadata, text_config)
        
    y_train, X_train_struc, X_train_text = encoder.fit_transform(df_train)
    y_dev, X_dev_struc, X_dev_text = encoder.transform(df_dev)
    y_test, X_test_struc, X_test_text = encoder.transform(df_test)

    if encoder.text_config is not None and encoder.text_config.mode == 'glove':
        f_path = os.path.join(path_to_save, 'embedding_matrix.npy')
        text_config.embedding_matrix_path = f_path
        with open(f_path, 'wb') as f:
            np.save(f, encoder.text_config.embedding_matrix)
        del encoder.text_config.embedding_matrix

    path = os.path.join(path_to_save, 'encoder.pkl')
    dump(encoder, open(path, 'wb'))

    metadata_path = os.path.join(path_to_save, 'metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4)

    if text_config is not None:
        text_config_path = os.path.join(path_to_save, 'text_config.json')
        with open(text_config_path, 'w') as f:
            json.dump(encoder.text_config, f, indent=4)
    
    
    ### save the encoded data ###
    output_list = [y_train, X_train_struc, X_train_text, y_dev, X_dev_struc, 
                   X_dev_text, y_test, X_test_struc, X_test_text]
    path_name_list = ['y_train', 'X_train_struc', 'X_train_text', 'y_dev', 'X_dev_struc', 
                      'X_dev_text', 'y_test', 'X_test_struc', 'X_test_text']

    for i, e in enumerate(output_list):
        if e is not None:
            e_path = os.path.join(path_to_save, '{}.npy'.format(path_name_list[i]))
            np.save(e_path, e)

    print('Saved the encoded text inputs!')


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


## use dict like object
class Mapping(dict):

    def __getattr__(self, name):
        if name in self:
            return self[name]
        else:
            raise AttributeError("No such attribute: " + name)

    def __setattr__(self, name, value):
        self[name] = value

    def __delattr__(self, name):
        if name in self:
            del self[name]
        else:
            raise AttributeError("No such attribute: " + name)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def contain_nontext_features(metadata):
    n_dtype = len(metadata.keys())

    if n_dtype == 1 and 'input_text' in metadata.keys():
        return False
    else:
        return True


def separate_input_output_cols(df, metadata):
    """According to the metadata, separate the input features, output features and 
        different types of input features.

    Args:
      df: a DataFrame that stores the raw data.
      metadata: a dictionary that stores the detail description for features.
        metadata = {
        'output_type': 'y', # for classification task (or it can be 'numbers' for regression task)
        'input_features': ['TenantId','CreatedDate', ...],
        'output_label': ['AR_exchange_06','AR_sharepoint_06', ...],
        'input_bool': ['HasEXO','HasSPO', ...],
        'input_categorical': ['CountryCode', 'Languange', ...],
        'input_datetime': ['CreatedDate', ...],
        'input_int': [...] ,
        'input_float': [...]
        }      
    Returns:
      df_y: a DataFrame that stores the output labels
      df_X_text: a DataFrame that stores the textual input
      df_X_float: a DataFrame that stores the float inputs
      df_X_int: a DataFrame that stores the integer inputs
      df_X_cat: a DataFrame that stores the categorical inputs
      df_X_datetime: a DataFrame that stores the datetime inputs
      df_X_bool: a DataFrame that stores the boolean inputs

    """
    # input_cols = metadata['input_features']
    output_cols = metadata['output_label']
    input_text_cols = metadata['input_text']
    input_float_cols = metadata['input_float']
    input_int_cols = metadata['input_int']
    input_cat_cols = metadata['input_categorical']
    input_datetime_cols = metadata['input_datetime']
    input_bool_cols = metadata['input_bool']

    df_y = df.loc[:, output_cols]
    df_X_text = df.loc[:, input_text_cols]
    df_X_float = df.loc[:, input_float_cols]
    df_X_int = df.loc[:, input_int_cols]
    df_X_cat = df.loc[:, input_cat_cols]
    df_X_datetime = df.loc[:, input_datetime_cols]
    df_X_bool = df.loc[:, input_bool_cols]

    return df_y, df_X_text, df_X_float, df_X_int, df_X_cat, df_X_datetime, df_X_bool


def encode_datetime(df_X_datetime):
    """Encode the datetime inputs from '2/5/2014 5:31:19 AM' format
        to a numerical number of UTC format.

    Args:
      df_: a DataFrame that only stores the datetime inputs.
        
    Returns:
      X_datetime: a numpy array that contains the encoded datetime inputs.
      datetime_cols: a list that contains the datetime colunms name.   
   
    """
    
    cols = df_X_datetime.columns
    for i in cols:
        df_X_datetime[i] = pd.to_datetime(df_X_datetime[i], utc=True,
                            errors='coerce').astype(int,errors='ignore')
        
    X_datetime = df_X_datetime.to_numpy()
    
    return X_datetime


def encode_bool(df_X_bool):
    """Encode the numerical and boolean inputs.
        
    Args:
      df_X_bool: a DataFrame that stores the boolean inputs
        
    Returns:
      X_bool: a numpy array that contains the encoded boolean inputs.

    """
    X_bool = df_X_bool.astype(int).to_numpy()
    return X_bool


def encode_num(df_X_num):
    """Encode the numerical and boolean inputs.
        
    Args:
      df_X_num: a DataFrame that stores the numerical inputs
        
    Returns:
      X_num: a numpy array that contains the float inputs.
      
    """
    X_num = df_X_num.to_numpy()
    return X_num


def encode_y(metadata, df_y, y_encoder):
    if metadata['output_type'] == 'classes':
        # encode class values as integers
        y_arr = df_y.values
        if y_encoder is None:
            y_encoder = LabelEncoder()
            y = y_encoder.fit_transform(y_arr)
        else:
            y = y_encoder.transform(y_arr)

        if len(y_encoder.classes_) > 2:
            # convert integers to dummy variables (i.e. one hot encoded)
            y = np_utils.to_categorical(y)

    elif metadata['output_type'] == 'numbers':
        y = df_y.to_numpy()
        y_encoder = None

    else:
        raise ValueError('Unknown type of output: {}'.format(metadata['output_type']))

    return y, y_encoder



def encode_strucdata(metadata, df_X_float, df_X_int, df_X_cat, df_X_datetime, df_X_bool, vectorizer, scaler):
    """Encode the meta data part in dataset, such as numerical and categorical data.
        
    """
    print('Starting to encode structural data...')

    # df_y, _, df_X_float, df_X_int, df_X_cat, df_X_datetime, df_X_bool = separate_input_output_cols(df, metadata)
    
    X_list = []
    cols_name = []
    
    if df_X_float.shape[1] > 0:
        X_float = encode_num(df_X_float)
        X_list.append(X_float)
        cols_name += metadata['input_float']

    if df_X_int.shape[1] > 0:
        X_int = encode_num(df_X_int)
        X_list.append(X_int)
        cols_name += metadata['input_int']
    
    if df_X_datetime.shape[1] > 0:
        X_datetime = encode_datetime(df_X_datetime)
        X_list.append(X_datetime)
        cols_name += metadata['input_datetime']

    if X_list:
        ### normalize all the inputs except boolean, categorical, and text features
        X_arr = np.concatenate(X_list, axis=1)

        if scaler == None:
            scaler = StandardScaler()
            X_struc = scaler.fit_transform(X_arr)
        else:
            X_struc = scaler.transform(X_arr)
        assert len(cols_name) == X_struc.shape[1]
        print('Except boolean, categorical and text input data after encoding, the shape is {}'.format(X_struc.shape))
        print('we have {} columns.'.format(len(cols_name)))
    else:
        X_struc = None

    ### encode boolean columns
    if df_X_bool.shape[1] > 0:
        X_bool = encode_bool(df_X_bool)
        cols_name += metadata['input_bool']
        if X_struc is None:
            X_struc = X_bool
        else:
            X_struc = np.concatenate([X_struc, X_bool], axis=1)

    ### encode the categorical columns 
    if df_X_cat.shape[1] > 0:
        X_cat_dict = df_X_cat.to_dict(orient='records')

        if vectorizer == None:   
            vectorizer = DictVectorizer(sparse=False)
            X_cat = vectorizer.fit_transform(X_cat_dict)
            
        else:
            X_cat = vectorizer.transform(X_cat_dict)

        vocab = vectorizer.vocabulary_
        vocab_od = collections.OrderedDict(sorted(vocab.items(), key=lambda x:x[1]))
        cat_encoded_cols = list(vocab_od.keys())
        cols_name += cat_encoded_cols
        if X_struc is None:
            X_struc = X_cat
        else:
            X_struc = np.concatenate([X_struc, X_cat], axis=1)

    assert len(cols_name) == X_struc.shape[1]
    print('Non-text input data after encoding, the shape is {}'.format(X_struc.shape))
    print('We have {} columns.'.format(len(cols_name)))
    
    return X_struc, vectorizer, scaler


def open_glove(glove_file_path):

    print('Indexing word vectors.')

    embeddings_index = {}
    f = open(glove_file_path)
    for line in f:
        word, coefs = line.split(maxsplit=1)
        coefs = np.fromstring(coefs, 'f', sep=' ')
        embeddings_index[word] = coefs
    f.close()

    print('Found %s word vectors.' % len(embeddings_index))

    return embeddings_index


def encode_textdata(df_X_text, tokenizer, mode, max_words, maxlen):
    ## encode text columns, encoded text features should not be normalized.

    print('Starting to encode text inputs...')

    texts = df_X_text.iloc[:,0].values.astype('U')
    print('Found %s texts.' % len(texts))

    if mode == 'tfidf':
        if tokenizer is None:
            tokenizer = Tokenizer(num_words=max_words)
            tokenizer.fit_on_texts(texts)
        X_text = tokenizer.texts_to_matrix(texts, mode='tfidf')
        print('tfidf X_text shape: {}'.format(X_text.shape))

    elif mode == 'glove':
        # vectorize the text samples into a 2D integer tensor
        if tokenizer is None:
            tokenizer = Tokenizer(num_words=max_words, oov_token='<UNK>')
            tokenizer.fit_on_texts(texts)
            tokenizer.word_index = {e:i for e,i in tokenizer.word_index.items() if i <= max_words}
            # tokenizer.word_index[tokenizer.oov_token] = max_words + 1

        sequences = tokenizer.texts_to_sequences(texts)
        
        word_index = tokenizer.word_index
        print('Found %s unique tokens.' % len(word_index))

        X_text = pad_sequences(sequences, maxlen=maxlen, padding='post')
    else:
        raise ValueError('Unknown text processing mode: {}'.format(mode))
                
    return X_text, tokenizer  ### need to save embedding_matrix as well


def encode_dataset(df, metadata, y_encoder=None, vectorizer=None, scaler=None, tokenizer=None, mode=None, 
                   max_words=None, maxlen=None):

    print('Starting to encode dataset...')

    df_y, df_X_text, df_X_float, df_X_int, df_X_cat, df_X_datetime, df_X_bool = separate_input_output_cols(df, metadata)

    y, y_encoder = encode_y(metadata, df_y, y_encoder)

    # check if exist non-text data
    if df_X_float.shape[1] + df_X_int.shape[1] + df_X_cat.shape[1] + df_X_datetime.shape[1] + df_X_bool.shape[1] > 0:
        X_struc, vectorizer, scaler = encode_strucdata(metadata, df_X_float, df_X_int, df_X_cat, df_X_datetime,
                                                       df_X_bool, vectorizer, scaler)
    else:
        X_struc, vectorizer, scaler = None, None, None

    print("complete encoding part of structural data!")

    if not metadata['input_text'] or mode == None:  
        X_text, tokenizer = None, None
    else:
        X_text, tokenizer = encode_textdata(df_X_text, tokenizer, mode, max_words, maxlen)

    print("complete encoding part of textual data!") 
    return y, y_encoder, X_struc, X_text, vectorizer, scaler, tokenizer


class Encoder(object):

    def __init__(self, metadata, text_config):
        self.text_config = text_config
        self.metadata = metadata
        self.has_nontext = contain_nontext_features(metadata)

    def fit_transform(self, df):
        if self.has_nontext and self.text_config is None:
            y, self.y_encoder, X_struc, X_text, self.vectorizer, self.scaler, _ = encode_dataset(df, self.metadata, mode=None)

        elif self.text_config.mode == 'tfidf':
            y, self.y_encoder, X_struc, X_text, self.vectorizer, self.scaler, self.tokenizer = encode_dataset(
                df, self.metadata, mode='tfidf', max_words=self.text_config.max_words)
            
        elif self.text_config.mode == 'glove':
            y, self.y_encoder, X_struc, X_text, self.vectorizer, self.scaler, self.tokenizer = encode_dataset(
                df, self.metadata, mode='glove', max_words=self.text_config.max_words, maxlen=self.text_config.maxlen)

            word_index = self.tokenizer.word_index
            # prepare embedding matrix
            embedding_matrix = np.zeros((len(word_index)+1, self.text_config.embedding_dim))
            for word, i in word_index.items():
                embedding_vector = self.text_config.embeddings_index.get(word)
                if embedding_vector is not None:
                    # words not found in embedding index will be all-zeros.
                    embedding_matrix[i] = embedding_vector
            self.text_config.embedding_matrix = embedding_matrix
            del self.text_config.embeddings_index

        else:
            raise ValueError('Unknown type of text_config: {}'.format(self.text_config.mode))
        
        return y, X_struc, X_text

    def transform(self, df):
        if self.text_config is None:
            y, _, X_struc, X_text, _, _, _ = encode_dataset(df, self.metadata, y_encoder=self.y_encoder,
                                                            vectorizer=self.vectorizer, scaler=self.scaler)

        elif self.text_config.mode == 'tfidf':

            y, _, X_struc, X_text, _, _, _ = encode_dataset(
                df, self.metadata, y_encoder=self.y_encoder,
                vectorizer=self.vectorizer, scaler=self.scaler, tokenizer=self.tokenizer, mode='tfidf',
                max_words=self.text_config.max_words)
        elif self.text_config.mode == 'glove':
            y, _, X_struc, X_text, _, _, _ = encode_dataset(
                df, self.metadata, y_encoder=self.y_encoder,
                vectorizer=self.vectorizer, scaler=self.scaler, tokenizer=self.tokenizer,
                mode='glove', max_words=self.text_config.max_words, maxlen=self.text_config.maxlen)
        else:
            raise ValueError('Unknown type of text_config: {}'.format(self.text_config.mode))

        return y, X_struc, X_text



if __name__ == '__main__':
    main()
