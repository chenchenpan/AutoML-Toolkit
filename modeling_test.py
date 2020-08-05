import os
import unittest
import numpy as np
from encoder import Encoder, Mapping, open_glove
from encoder_test import get_fake_dataset
from modeling import LogisticRegressionModel, SVMModel, RandomForestModel, NeuralNetworkModel, LinearRegressionModel


def get_fake_modelconfig(output_path):
    model_config = Mapping()
    model_config.task_type = 'classification' ## 'classification' or 'regression'
    model_config.num_classes = 3 ## number of classes or number of outputs
    model_config.combine = 'concate' ## or 'attention'
    model_config.model_type = 'mlp' ## default is 'mlp', can be 'skip_connections'
    model_config.n_layers_dense = 2
    model_config.hidden_size_dense = 16
    model_config.n_layers_lstm = 2
    model_config.hidden_size_lstm = 32
    model_config.dropout_rate_lstm = 0.0
    model_config.n_layers_output = 2
    model_config.hidden_size_output = 32
    model_config.optimizer = 'adam' ## 'adam', 'sgd', 'rmsprop'
    model_config.learning_rate = 0.01
    model_config.clipnorm = 5.0
    model_config.patience = 5
    model_config.output_dir = output_path
    model_config.n_epochs = 10
    model_config.batch_size = 1
    model_config.verbose = 0
    return model_config


def get_fake_lr_modelconfig(output_path):
    model_config = Mapping()
    model_config.task_type = 'classification' ## 'classification' or 'regression'
    model_config.num_classes = 3 ## number of classes or number of outputs
    model_config.model_type = 'logistic_regression' ## default is 'mlp', can be 'skip_connections'
    model_config.output_dir = output_path
    model_config.C = 0.1
    return model_config


def get_fake_rf_modelconfig(output_path):
    model_config = Mapping()
    model_config.task_type = 'classification' ## 'classification' or 'regression'
    model_config.num_classes = 3 ## number of classes or number of outputs
    model_config.model_type = 'random_forest' ## default is 'mlp', can be 'skip_connections'
    model_config.output_dir = output_path
    model_config.n_trees = 4
    return model_config


def get_fake_svm_modelconfig(output_path):
    model_config = Mapping()
    model_config.task_type = 'classification' ## 'classification' or 'regression'
    model_config.num_classes = 3 ## number of classes or number of outputs
    model_config.model_type = 'svm' ## default is 'mlp', can be 'skip_connections'
    model_config.output_dir = output_path
    model_config.C = 0.1
    return model_config


def get_fake_linear_regression_modelconfig(output_path):
    model_config = Mapping()
    model_config.task_type = 'regression' ## 'classification' or 'regression'
    model_config.num_classes = 3 ## number of classes or number of outputs
    model_config.model_type = 'linear_regression' ## default is 'mlp', can be 'skip_connections'
    model_config.output_dir = output_path
    # model_config.C = 0.1
    return model_config


class TestModel(unittest.TestCase):
    def test_lstm(self):
        df_train, df_dev, df_test, metadata = get_fake_dataset(with_text_col=True)

        glove_file_path = 'glove/glove.6B.50d.txt'# need be changed to where you store the pre-trained GloVe file.
        
        text_config = Mapping()
        text_config.mode = 'glove'
        text_config.max_words = 20
        text_config.maxlen = 5
        text_config.embedding_dim = 50
        text_config.embeddings_index = open_glove(glove_file_path) # need to change

        encoder = Encoder(metadata, text_config=text_config)
        y_train, X_train_struc, X_train_text = encoder.fit_transform(df_train)
        y_dev, X_dev_struc, X_dev_text = encoder.transform(df_dev)
        y_test, X_test_struc, X_test_text = encoder.transform(df_test)

        text_config.embedding_matrix = encoder.embedding_matrix

        model_config = get_fake_modelconfig('./outputs_test')
        model_config.output_dir = os.path.join(model_config.output_dir, 'lstm')
        if not os.path.exists(model_config.output_dir):
            os.makedirs(model_config.output_dir)

        model = NeuralNetworkModel(text_config, model_config)
        output = model.train(y_train, X_train_struc, X_train_text, y_train, X_train_struc, X_train_text)

        # print(hist.history)
        # y_dev, X_dev_struc, X_dev_text)

        val_metric_true = 0.0
        self.assertTrue(np.isclose(val_metric_true, output['val_metric'], atol=1e-4))



    def test_tfidf(self):
        df_train, df_dev, df_test, metadata = get_fake_dataset(with_text_col=True)
        
        text_config = Mapping()
        text_config.mode = 'tfidf'
        text_config.max_words = 20

        encoder = Encoder(metadata, text_config)
        y_train, X_train_struc, X_train_text = encoder.fit_transform(df_train)
        y_dev, X_dev_struc, X_dev_text = encoder.transform(df_dev)
        y_test, X_test_struc, X_test_text = encoder.transform(df_test)

        model_config = get_fake_modelconfig('./outputs_test')
        model_config.output_dir = os.path.join(model_config.output_dir, 'tfidf_text_only')
        if not os.path.exists(model_config.output_dir):
            os.makedirs(model_config.output_dir)

        model = NeuralNetworkModel(text_config, model_config)
        output = model.train(y_train, X_train_struc, X_train_text, y_train, X_train_struc, X_train_text)

        # print(hist.history)
        # y_dev, X_dev_struc, X_dev_text)

        val_metric_true = 0.0
        self.assertTrue(np.isclose(val_metric_true, output['val_metric']))



    def test_strucdata_only(self):
        df_train, df_dev, df_test, metadata = get_fake_dataset(with_text_col=False)
        encoder = Encoder(metadata, text_config=None)
        y_train, X_train_struc, X_train_text = encoder.fit_transform(df_train)
        y_dev, X_dev_struc, X_dev_text = encoder.transform(df_dev)
        y_test, X_test_struc, X_test_text = encoder.transform(df_test)

        model_config = get_fake_modelconfig('./outputs_test')
        model_config.output_dir = os.path.join(model_config.output_dir, 'dense_mlp')
        if not os.path.exists(model_config.output_dir):
            os.makedirs(model_config.output_dir)

        model = NeuralNetworkModel(text_config=None, model_config=model_config)
        output = model.train(y_train, X_train_struc, X_train_text, y_train, X_train_struc, X_train_text)

        # print(hist.history)
        # y_dev, X_dev_struc, X_dev_text)

        val_metric_true = 0.0
        self.assertTrue(np.isclose(val_metric_true, output['val_metric'], atol=1e-2))



    def test_textdata_only_glove(self):
        df_train, df_dev, df_test, metadata = get_fake_dataset(with_text_col=True, text_only=True)

        glove_file_path = 'glove/glove.6B.50d.txt'# need be changed to where you store the pre-trained GloVe file.
        
        text_config = Mapping()
        text_config.mode = 'glove'
        text_config.max_words = 20
        text_config.maxlen = 5
        text_config.embedding_dim = 50
        text_config.embeddings_index = open_glove(glove_file_path) # need to change

        encoder = Encoder(metadata, text_config=text_config)
        y_train, X_train_struc, X_train_text = encoder.fit_transform(df_train)
        y_dev, X_dev_struc, X_dev_text = encoder.transform(df_dev)
        y_test, X_test_struc, X_test_text = encoder.transform(df_test)

        text_config.embedding_matrix = encoder.embedding_matrix

        model_config = get_fake_modelconfig('./outputs_test')
        model_config.output_dir = os.path.join(model_config.output_dir, 'lstm_text_only')
        if not os.path.exists(model_config.output_dir):
            os.makedirs(model_config.output_dir)

        model = NeuralNetworkModel(text_config, model_config)
        output = model.train(y_train, X_train_struc, X_train_text, y_train, X_train_struc, X_train_text)

        # print(hist.history)
        # y_dev, X_dev_struc, X_dev_text)

        val_metric_true = 0.0
        print(output['val_metric'])
        self.assertTrue(np.isclose(val_metric_true, output['val_metric']))


    def test_textdata_only_tfidf(self):
        df_train, df_dev, df_test, metadata = get_fake_dataset(with_text_col=True, text_only=True)
        
        text_config = Mapping()
        text_config.mode = 'tfidf'
        text_config.max_words = 20

        encoder = Encoder(metadata, text_config=text_config)
        y_train, X_train_struc, X_train_text = encoder.fit_transform(df_train)
        y_dev, X_dev_struc, X_dev_text = encoder.transform(df_dev)
        y_test, X_test_struc, X_test_text = encoder.transform(df_test)

        model_config = get_fake_modelconfig('./outputs_test')
        model_config.output_dir = os.path.join(model_config.output_dir, 'tfidf_text_only')
        if not os.path.exists(model_config.output_dir):
            os.makedirs(model_config.output_dir)

        model = NeuralNetworkModel(text_config, model_config)
        output = model.train(y_train, X_train_struc, X_train_text, y_train, X_train_struc, X_train_text)

        # print(hist.history)
        # y_dev, X_dev_struc, X_dev_text)

        val_metric_true = 0.0
        self.assertTrue(np.isclose(val_metric_true, output['val_metric']))

    def test_logistic_regression(self):
        df_train, df_dev, df_test, metadata = get_fake_dataset(with_text_col=True, text_only=True)
        
        text_config = Mapping()
        text_config.mode = 'tfidf'
        text_config.max_words = 20

        encoder = Encoder(metadata, text_config=text_config)
        y_train, X_train_struc, X_train_text = encoder.fit_transform(df_train)
        y_dev, X_dev_struc, X_dev_text = encoder.transform(df_dev)
        y_test, X_test_struc, X_test_text = encoder.transform(df_test)

        model_config = get_fake_lr_modelconfig('./outputs_test')
        model_config.output_dir = os.path.join(model_config.output_dir, 'tfidf_text_only')
        if not os.path.exists(model_config.output_dir):
            os.makedirs(model_config.output_dir)

        model = LogisticRegressionModel(text_config, model_config)
        output = model.train(y_train, X_train_struc, X_train_text, y_train, X_train_struc, X_train_text)

        val_metric_true = 0.0
        self.assertTrue(np.isclose(val_metric_true, output['val_metric']))

    def test_svm(self):
        df_train, df_dev, df_test, metadata = get_fake_dataset(with_text_col=True, text_only=True)
        
        text_config = Mapping()
        text_config.mode = 'tfidf'
        text_config.max_words = 20

        encoder = Encoder(metadata, text_config=text_config)
        y_train, X_train_struc, X_train_text = encoder.fit_transform(df_train)
        y_dev, X_dev_struc, X_dev_text = encoder.transform(df_dev)
        y_test, X_test_struc, X_test_text = encoder.transform(df_test)

        model_config = get_fake_svm_modelconfig('./outputs_test')
        model_config.output_dir = os.path.join(model_config.output_dir, 'tfidf_text_only')
        if not os.path.exists(model_config.output_dir):
            os.makedirs(model_config.output_dir)

        model = SVMModel(text_config, model_config)
        output = model.train(y_train, X_train_struc, X_train_text, y_train, X_train_struc, X_train_text)

        val_metric_true = 0.0
        self.assertTrue(np.isclose(val_metric_true, output['val_metric']))

    def test_random_forest(self):
        df_train, df_dev, df_test, metadata = get_fake_dataset(with_text_col=True, text_only=True)
        
        text_config = Mapping()
        text_config.mode = 'tfidf'
        text_config.max_words = 20

        encoder = Encoder(metadata, text_config=text_config)
        y_train, X_train_struc, X_train_text = encoder.fit_transform(df_train)
        y_dev, X_dev_struc, X_dev_text = encoder.transform(df_dev)
        y_test, X_test_struc, X_test_text = encoder.transform(df_test)

        model_config = get_fake_rf_modelconfig('./outputs_test')
        model_config.output_dir = os.path.join(model_config.output_dir, 'tfidf_text_only')
        if not os.path.exists(model_config.output_dir):
            os.makedirs(model_config.output_dir)

        model = RandomForestModel(text_config, model_config)
        output = model.train(y_train, X_train_struc, X_train_text, y_train, X_train_struc, X_train_text)

        val_metric_true = 0.0
        self.assertTrue(np.isclose(val_metric_true, output['val_metric']))

    
    def test_linear_regression(self):
        df_train, df_dev, df_test, metadata = get_fake_dataset(with_text_col=True, text_only=True)
        
        text_config = Mapping()
        text_config.mode = 'tfidf'
        text_config.max_words = 20

        encoder = Encoder(metadata, text_config=text_config)
        y_train, X_train_struc, X_train_text = encoder.fit_transform(df_train)
        y_dev, X_dev_struc, X_dev_text = encoder.transform(df_dev)
        y_test, X_test_struc, X_test_text = encoder.transform(df_test)

        model_config = get_fake_linear_regression_modelconfig('./outputs_test')
        model_config.output_dir = os.path.join(model_config.output_dir, 'tfidf_text_only')
        if not os.path.exists(model_config.output_dir):
            os.makedirs(model_config.output_dir)

        model = LinearRegressionModel(text_config, model_config)
        output = model.train(y_train, X_train_struc, X_train_text, y_train, X_train_struc, X_train_text)

        val_metric_true = 0.0
        self.assertTrue(np.isclose(val_metric_true, output['val_metric']))



    def test_skip_connections(self):
        pass



if __name__ == '__main__':
    unittest.main()



# ModelConfig = {
#     'task_type': {'value': 'classification', 'type': 'fixed'},
#     'num_outputs': {'value': 2, 'type': 'fixed'},
#     ''
# }


# self.dropout_rate = dropout_rate
#         self.n_lstm_layers = n_lstm_layers
#         self.lstm_hidden_size = lstm_hidden_size
#         self.hidden_size = hidden_size
#         self.n_layers = n_layers
#         self.max_n_epoch = max_n_epoch
#         self.patience = patience
#         self.learning_rate = learning_rate
#         self.useL2 = use_L2regularizer
#         self.batch_size = batch_size
#         self.opt = optimizer

# TextConfig = {
    
# }

# python preprocessing.y --text_config=""


# preprocessed_with_tfidf/text_config.json


# python experiment.py --data_dir=preprocessed_with_glove 




