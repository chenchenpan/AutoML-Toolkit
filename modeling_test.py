import os
import unittest
import numpy as np
import pandas as pd
from encode_data import Encoder, Mapping, open_glove
from encode_data_test import get_fake_dataset, get_fake_dataset_binary_class
from modeling import LogisticRegressionModel, SVMModel, RandomForestModel, NeuralNetworkModel, LinearRegressionModel


def get_fake_modelconfig(output_path):
    model_config = Mapping()
    model_config.task_type = 'classification' ## 'classification' or 'regression'
    model_config.metric = 'acc'
    model_config.num_classes = 3 ## number of classes
    model_config.num_outputs = 1 ## or number of outputs
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
    model_config.metric = 'acc'
    model_config.num_classes = 3 ## number of classes
    model_config.num_outputs = 1  ## or number of outputs
    model_config.model_type = 'logistic_regression' ## default is 'mlp', can be 'skip_connections'
    model_config.output_dir = output_path
    model_config.C = 0.1
    return model_config


def get_fake_rf_modelconfig(output_path):
    model_config = Mapping()
    model_config.task_type = 'classification' ## 'classification' or 'regression'
    model_config.metric = 'acc'
    model_config.num_classes = 3 ## number of classes
    model_config.num_outputs = 1  ## or number of outputs
    model_config.model_type = 'random_forest' ## default is 'mlp', can be 'skip_connections'
    model_config.output_dir = output_path
    model_config.n_trees = 4
    return model_config


def get_fake_svm_modelconfig(output_path):
    model_config = Mapping()
    model_config.task_type = 'classification' ## 'classification' or 'regression'
    model_config.metric = 'acc'
    model_config.num_classes = 3 ## number of classes
    model_config.num_outputs = 1  ## or number of outputs
    model_config.model_type = 'svm' ## default is 'mlp', can be 'skip_connections'
    model_config.output_dir = output_path
    model_config.C = 0.1
    return model_config


def get_fake_linear_regression_modelconfig(output_path):
    model_config = Mapping()
    model_config.task_type = 'regression' ## 'classification' or 'regression'
    model_config.metric = 'mse'
    model_config.num_classes = 3 ## number of classes
    model_config.num_outputs = 1  ## or number of outputs
    model_config.model_type = 'linear_regression' ## default is 'mlp', can be 'skip_connections'
    model_config.output_dir = output_path
    # model_config.C = 0.1
    return model_config


class TestModel(unittest.TestCase):
    def test_auc_metric_on_random_forest(self):
        df_train, df_dev, df_test, metadata = get_fake_dataset_binary_class(with_text_col=True, text_only=True)

        text_config = Mapping()
        text_config.mode = 'tfidf'
        text_config.max_words = 20

        encoder = Encoder(metadata, text_config=text_config)
        y_train, X_train_struc, X_train_text = encoder.fit_transform(df_train)
        y_dev, X_dev_struc, X_dev_text = encoder.transform(df_dev)
        y_test, X_test_struc, X_test_text = encoder.transform(df_test)

        model_config = get_fake_rf_modelconfig('tmp/outputs_test')
        model_config.num_classes = 2
        model_config.metric = 'auc'
        model_config.output_dir = os.path.join(model_config.output_dir, 'rf_tfidf_text_only_auc')
        if not os.path.exists(model_config.output_dir):
            os.makedirs(model_config.output_dir)

        model = RandomForestModel(text_config, model_config)
        output = model.train(y_train, X_train_struc, X_train_text, y_train, X_train_struc, X_train_text)

        print(y_train)
        # output = model.train(y_train, X_train_struc, X_train_text, y_train, X_train_struc, X_train_text)

        val_metric_true = 0.0
        self.assertTrue(np.isclose(val_metric_true, output['val_metric']))


    def test_auc_metric_on_logistic_regression(self):
        df_train, df_dev, df_test, metadata = get_fake_dataset_binary_class(with_text_col=False)

        # text_config = Mapping()
        # text_config.mode = 'tfidf'
        # text_config.max_words = 20

        encoder = Encoder(metadata, text_config=None)
        y_train, X_train_struc, X_train_text = encoder.fit_transform(df_train)
        y_dev, X_dev_struc, X_dev_text = encoder.transform(df_dev)
        y_test, X_test_struc, X_test_text = encoder.transform(df_test)

        model_config = get_fake_lr_modelconfig('tmp/outputs_test')
        model_config.num_classes = 2
        model_config.metric = 'auc'
        model_config.output_dir = os.path.join(model_config.output_dir, 'logistic_regression_auc')
        if not os.path.exists(model_config.output_dir):
            os.makedirs(model_config.output_dir)

        model = LogisticRegressionModel(text_config=None, model_config=model_config)
        output = model.train(y_train, X_train_struc, X_train_text, y_train, X_train_struc, X_train_text)

        val_metric_true = 0.0
        print(output['val_metric'])
        self.assertTrue(np.isclose(val_metric_true, output['val_metric']))

    # the auc metric cannot work for binary classification task with NN model
    def test_auc_metric_on_mlp(self):
        df_train, df_dev, df_test, metadata = get_fake_dataset_binary_class(with_text_col=False)
        encoder = Encoder(metadata, text_config=None)
        y_train, X_train_struc, X_train_text = encoder.fit_transform(df_train)
        y_dev, X_dev_struc, X_dev_text = encoder.transform(df_dev)
        y_test, X_test_struc, X_test_text = encoder.transform(df_test)

        model_config = get_fake_modelconfig('tmp/outputs_test')
        model_config.num_classes = 2
        model_config.metric = 'auc'
        print('*' * 20)
        print('model_config is {}'.format(model_config))
        # print(output)
        print('*' * 20)

        model_config.output_dir = os.path.join(model_config.output_dir, 'dense_mlp_binary_auc')
        if not os.path.exists(model_config.output_dir):
            os.makedirs(model_config.output_dir)

        model = NeuralNetworkModel(text_config=None, model_config=model_config)
        output = model.train(y_train, X_train_struc, X_train_text, y_train, X_train_struc, X_train_text)

        # print('*' * 20)
        # print(output['val_metric'])
        # # y_dev, X_dev_struc, X_dev_text)
        # print('*' * 20)
        print('*' * 20)
        print(model.hist.history)
        # print(output)
        print('*' * 20)

        val_metric_true = 0.0
        self.assertTrue(np.isclose(val_metric_true, output['val_metric'], atol=1e-2))

    # # the metric tf.keras.metrics.AUC() cannot work for multiclass classification task with SparseCategoricalCrossentropy
    # # some work-around can be find in https://stackoverflow.com/questions/69357626/incompatible-dimension-when-using-sparsecategoricalentropy-loss-in-keras

    # def test_auc_metric_on_multiclass_classification(self):
    #     df_train, df_dev, df_test, metadata = get_fake_dataset(with_text_col=False)
    #     encoder = Encoder(metadata, text_config=None)
    #     y_train, X_train_struc, X_train_text = encoder.fit_transform(df_train)
    #     y_dev, X_dev_struc, X_dev_text = encoder.transform(df_dev)
    #     y_test, X_test_struc, X_test_text = encoder.transform(df_test)
    #
    #     model_config = get_fake_modelconfig('tmp/outputs_test')
    #     model_config.metric = 'auc'
    #     print('*' * 20)
    #     print('model_config is {}'.format(model_config))
    #     # print(output)
    #     print('*' * 20)
    #
    #     model_config.output_dir = os.path.join(model_config.output_dir, 'dense_mlp_auc')
    #     if not os.path.exists(model_config.output_dir):
    #         os.makedirs(model_config.output_dir)
    #
    #     model = NeuralNetworkModel(text_config=None, model_config=model_config)
    #     output = model.train(y_train, X_train_struc, X_train_text, y_train, X_train_struc, X_train_text)
    #
    #     # print('*' * 20)
    #     # print(output['val_metric'])
    #     # # y_dev, X_dev_struc, X_dev_text)
    #     # print('*' * 20)
    #     print('*' * 20)
    #     print(model.hist.history)
    #     # print(output)
    #     print('*' * 20)
    #
    #     val_metric_true = 0.0
    #     self.assertTrue(np.isclose(val_metric_true, output['val_metric'], atol=1e-2))


    def test_lstm(self):
        df_train, df_dev, df_test, metadata = get_fake_dataset(with_text_col=True)

        glove_file_path = 'resource/glove/glove.6B.50d.txt'# need be changed to where you store the pre-trained GloVe file.

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

        text_config.embedding_matrix = encoder.text_config.embedding_matrix

        model_config = get_fake_modelconfig('tmp/outputs_test')
        model_config.output_dir = os.path.join(model_config.output_dir, 'lstm')
        if not os.path.exists(model_config.output_dir):
            os.makedirs(model_config.output_dir)

        model = NeuralNetworkModel(text_config, model_config)
        output = model.train(y_train, X_train_struc, X_train_text, y_train, X_train_struc, X_train_text)

        print('*' * 20)
        print(model.hist.history)
        print(output)
        print('*' * 20)

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

        model_config = get_fake_modelconfig('tmp/outputs_test')
        model_config.output_dir = os.path.join(model_config.output_dir, 'tfidf_text_only')
        if not os.path.exists(model_config.output_dir):
            os.makedirs(model_config.output_dir)

        model = NeuralNetworkModel(text_config, model_config)
        output = model.train(y_train, X_train_struc, X_train_text, y_train, X_train_struc, X_train_text)

        print(model.hist.history)
        print(y_dev, X_dev_struc, X_dev_text)

        val_metric_true = 0.0
        self.assertTrue(np.isclose(val_metric_true, output['val_metric']))



    def test_strucdata_only(self):
        df_train, df_dev, df_test, metadata = get_fake_dataset(with_text_col=False)
        encoder = Encoder(metadata, text_config=None)
        y_train, X_train_struc, X_train_text = encoder.fit_transform(df_train)
        y_dev, X_dev_struc, X_dev_text = encoder.transform(df_dev)
        y_test, X_test_struc, X_test_text = encoder.transform(df_test)

        model_config = get_fake_modelconfig('tmp/outputs_test')
        model_config.output_dir = os.path.join(model_config.output_dir, 'dense_mlp')
        if not os.path.exists(model_config.output_dir):
            os.makedirs(model_config.output_dir)

        model = NeuralNetworkModel(text_config=None, model_config=model_config)
        output = model.train(y_train, X_train_struc, X_train_text, y_train, X_train_struc, X_train_text)

        print('*' * 20)
        print(output['val_metric'])
        # y_dev, X_dev_struc, X_dev_text)
        print('*' * 20)

        val_metric_true = 0.0
        self.assertTrue(np.isclose(val_metric_true, output['val_metric'], atol=1e-2))



    def test_textdata_only_glove(self):
        df_train, df_dev, df_test, metadata = get_fake_dataset(with_text_col=True, text_only=True)

        glove_file_path = 'resource/glove/glove.6B.50d.txt'# need be changed to where you store the pre-trained GloVe file.

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

        text_config.embedding_matrix = encoder.text_config.embedding_matrix

        model_config = get_fake_modelconfig('tmp/outputs_test')
        model_config.output_dir = os.path.join(model_config.output_dir, 'lstm_text_only')
        if not os.path.exists(model_config.output_dir):
            os.makedirs(model_config.output_dir)

        model = NeuralNetworkModel(text_config, model_config)
        output = model.train(y_train, X_train_struc, X_train_text, y_train, X_train_struc, X_train_text)

        # print(hist.history)
        # y_dev, X_dev_struc, X_dev_text)

        val_metric_true = 0.0
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

        model_config = get_fake_modelconfig('tmp/outputs_test')
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

        model_config = get_fake_lr_modelconfig('tmp/outputs_test')
        model_config.output_dir = os.path.join(model_config.output_dir, 'tfidf_text_only')
        if not os.path.exists(model_config.output_dir):
            os.makedirs(model_config.output_dir)

        model = LogisticRegressionModel(text_config, model_config)

        print(y_train)
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

        model_config = get_fake_svm_modelconfig('tmp/outputs_test')
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

        model_config = get_fake_rf_modelconfig('tmp/outputs_test')
        model_config.output_dir = os.path.join(model_config.output_dir, 'rf_tfidf_text_only')
        if not os.path.exists(model_config.output_dir):
            os.makedirs(model_config.output_dir)

        model = RandomForestModel(text_config, model_config)
        output = model.train(y_train, X_train_struc, X_train_text, y_train, X_train_struc, X_train_text)

        print(y_train)
        # output = model.train(y_train, X_train_struc, X_train_text, y_train, X_train_struc, X_train_text)

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

        model_config = get_fake_linear_regression_modelconfig('tmp/outputs_test')
        model_config.output_dir = os.path.join(model_config.output_dir, 'tfidf_text_only')
        if not os.path.exists(model_config.output_dir):
            os.makedirs(model_config.output_dir)

        model = LinearRegressionModel(text_config, model_config)
        output = model.train(y_train, X_train_struc, X_train_text, y_train, X_train_struc, X_train_text)

        val_metric_true = 0.0
        print(output['val_metric'])
        self.assertTrue(np.isclose(val_metric_true, output['val_metric']))



    def test_skip_connections(self):
        pass

    def test_multi_task_learning(self):
        """Test multi-task learning with classification and regression tasks"""
        # Generate fake dataset with multiple targets
        df_train, df_dev, df_test, metadata = get_fake_dataset_multi_task(
            with_text_col=True,
            classification_targets=['sentiment'],  # Binary classification task
            regression_targets=['rating']          # Regression task
        )
        
        # Configure text processing
        text_config = Mapping()
        text_config.mode = 'tfidf'
        text_config.max_words = 20
        
        # Create encoder and transform data
        encoder = Encoder(metadata, text_config=text_config)
        y_train_dict, X_train_struc, X_train_text = encoder.fit_transform_multi_task(df_train)
        y_dev_dict, X_dev_struc, X_dev_text = encoder.transform_multi_task(df_dev)
        
        # Configure model for multi-task learning
        model_config = get_fake_modelconfig('tmp/outputs_test')
        model_config.update({
            'task_types': ['classification', 'regression'],
            'task_names': ['sentiment', 'rating'],
            'num_classes_list': [2, None],  # Binary classification and regression
            'metric': 'auc',                # Primary metric for classification
            'task_specific_layers': 2,      # Add task-specific layers
            'hidden_size_output': 64
        })
        
        # Set output directory
        model_config.output_dir = os.path.join(model_config.output_dir, 'multi_task')
        if not os.path.exists(model_config.output_dir):
            os.makedirs(model_config.output_dir)
        
        # Create and train model
        model = NeuralNetworkModel(text_config, model_config)
        val_metrics = model.train_multi_task(
            y_train_dict, X_train_struc, X_train_text,
            y_dev_dict, X_dev_struc, X_dev_text
        )
        
        # Check if expected metrics are returned
        self.assertIn('sentiment_error_rate', val_metrics)
        self.assertIn('rating_mse', val_metrics)
        
        # Check if metrics are within expected range
        self.assertGreaterEqual(val_metrics['sentiment_error_rate'], 0.0)
        self.assertLessEqual(val_metrics['sentiment_error_rate'], 1.0)
        self.assertGreaterEqual(val_metrics['rating_mse'], 0.0)

    def test_multi_task_learning_classification_only(self):
        """Test multi-task learning with multiple classification tasks"""
        # Generate fake dataset with multiple classification targets
        df_train, df_dev, df_test, metadata = get_fake_dataset_multi_task(
            with_text_col=True,
            classification_targets=['sentiment', 'topic'],  # Binary and multi-class
            regression_targets=[]
        )
        
        # Configure text processing
        text_config = Mapping()
        text_config.mode = 'tfidf'
        text_config.max_words = 20
        
        # Create encoder and transform data
        encoder = Encoder(metadata, text_config=text_config)
        y_train_dict, X_train_struc, X_train_text = encoder.fit_transform_multi_task(df_train)
        y_dev_dict, X_dev_struc, X_dev_text = encoder.transform_multi_task(df_dev)
        
        # Configure model for multi-task classification
        model_config = get_fake_modelconfig('tmp/outputs_test')
        model_config.task_types = ['classification', 'classification']
        model_config.task_names = ['sentiment', 'topic']
        model_config.num_classes_list = [2, 3]  # Binary and 3-class classification
        model_config.metric = 'acc'
        model_config.task_specific_layers = 1
        model_config.output_dir = os.path.join(model_config.output_dir, 'multi_task_classification')
        
        if not os.path.exists(model_config.output_dir):
            os.makedirs(model_config.output_dir)
        
        # Create and train model
        model = NeuralNetworkModel(text_config, model_config)
        val_metrics = model.train_multi_task(
            y_train_dict, X_train_struc, X_train_text,
            y_dev_dict, X_dev_struc, X_dev_text
        )
        
        # Check metrics
        self.assertIn('sentiment_error_rate', val_metrics)
        self.assertIn('topic_error_rate', val_metrics)
        
        # Verify metric ranges
        for task in ['sentiment', 'topic']:
            self.assertGreaterEqual(val_metrics[f'{task}_error_rate'], 0.0)
            self.assertLessEqual(val_metrics[f'{task}_error_rate'], 1.0)


def get_fake_dataset_multi_task(with_text_col=True, classification_targets=None, regression_targets=None):
    """Generate fake dataset for multi-task learning"""
    n_samples = 100
    data = {
        'id': [f'{i:02d}' for i in range(n_samples)],
        'float_col': np.random.randn(n_samples),
        'int_col': np.random.randint(0, 5, n_samples),
        'categorical_col': np.random.choice(['A', 'B', 'C'], n_samples)
    }
    
    if with_text_col:
        data['text_col'] = [
            'Sample text ' + str(i) for i in range(n_samples)
        ]
    
    # Add classification targets
    if classification_targets:
        for target in classification_targets:
            if target == 'sentiment':  # Binary
                data[target] = np.random.randint(0, 2, n_samples)
            else:  # Multi-class
                data[target] = np.random.randint(0, 3, n_samples)
    
    # Add regression targets
    if regression_targets:
        for target in regression_targets:
            data[target] = np.random.randn(n_samples)
    
    # Create DataFrame and split
    df = pd.DataFrame(data)
    train_size = int(0.6 * len(df))
    dev_size = int(0.2 * len(df))
    
    df_train = df[:train_size]
    df_dev = df[train_size:train_size + dev_size]
    df_test = df[train_size + dev_size:]
    
    # Create metadata
    metadata = {
        'input_features': ['float_col', 'int_col', 'categorical_col'],
        'output_label': classification_targets + (regression_targets if regression_targets else []),
        'input_float': ['float_col'],
        'input_int': ['int_col'],
        'input_categorical': ['categorical_col'],
        'input_datetime': [],
        'input_bool': [],
        'input_text': ['text_col'] if with_text_col else [],
        'output_type': 'multi_task',
        'task_types': {
            target: 'classification' for target in (classification_targets or [])
        } | {
            target: 'regression' for target in (regression_targets or [])
        }
    }
    
    return df_train, df_dev, df_test, metadata



if __name__ == '__main__':
    if not os.path.exists('tmp'):
        os.makedirs('tmp')    
    unittest.main()
