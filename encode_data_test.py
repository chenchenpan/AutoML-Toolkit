import os
import unittest
from encode_data import Encoder, Mapping, open_glove
import pandas as pd
import numpy as np
import json

def get_fake_dataset(with_text_col=False, text_only=False, output_type='classes'):
## you can change this to create your own test dataset here ##
    if output_type == 'classes':
        if with_text_col:
            df_train = pd.DataFrame({'height': [1,2,3], 'key_words': ['hello', 'hi', 'yes'],
                             'text': ["Strange Wit, an original graphic novel about Jane Bowles",
                                      "The true biography of the historical figure, writer, alcoholic, lesbian",
                                      "world traveler: Jane Sydney Auer Bowles."],
                             'label': [0, 1, 2]})
            df_dev = pd.DataFrame({'height': [4,7,5], 'key_words': ['hi', 'hi', 'yes'],
                                   'text': ["FAM is the new mobile app which combines events and all your social media needs",
                                            "Destiny, NY - FINAL HOURS!",
                                            "A graphic novel about two magical ladies in love."],
                                   'label': [1, 1, 2]})
            df_test = pd.DataFrame({'height': [2,5,3], 'key_words': ['hello', 'yes', 'yes'],
                            'text':["Publishing Magus Magazine,We are publishing a magazine that focuses on the folklore of the occult and paranormal.",
                                    "It is tabloid format but with academic articles",
                                    "a strong-willed Russian madam and The Cross at its most fabulous."],
                            'label': [2, 1, 2]})
            if text_only:
                metadata = {'output_type': 'classes',
                            'input_features': ['text'],
                            'output_label': ['label'],
                            'input_text': ['text'],
                            'input_bool': [],
                            'input_categorical': [],
                            'input_datetime': [],
                            'input_int': [],
                            'input_float': []
                            }

            else:
                metadata = {'output_type': 'classes',
                            'input_features': ['height','key_words','text'],
                            'output_label': ['label'],
                            'input_text': ['text'],
                            'input_bool': [],
                            'input_categorical': ['key_words'],
                            'input_datetime': [],
                            'input_int': ['height'],
                            'input_float': []
                            }
        else:
            df_train = pd.DataFrame({'height': [1,2,3], 'key_words': ['hello', 'hi', 'yes'], 'label': [0, 1, 2]})
            df_dev = pd.DataFrame({'height': [4,7,5], 'key_words': ['hi', 'hi', 'yes'], 'label': [1, 1, 2]})
            df_test = pd.DataFrame({'height': [2,5,3], 'key_words': ['hello', 'yes', 'yes'], 'label': [2, 1, 2]})
            metadata = {'output_type': 'classes',
                        'input_features': ['height','key_words'],
                        'output_label': ['label'],
                        'input_text': [],
                        'input_bool': [],
                        'input_categorical': ['key_words'],
                        'input_datetime': [],
                        'input_int': ['height'],
                        'input_float': []
                        }
    elif output_type == 'numbers':
        if with_text_col:
            df_train = pd.DataFrame({'height': [1, 2, 3], 'key_words': ['hello', 'hi', 'yes'],
                                     'text': ["Strange Wit, an original graphic novel about Jane Bowles",
                                              "The true biography of the historical figure, writer, alcoholic, lesbian",
                                              "world traveler: Jane Sydney Auer Bowles."],
                                     'label': [0, 1, 2]})
            df_dev = pd.DataFrame({'height': [4, 7, 5], 'key_words': ['hi', 'hi', 'yes'],
                                   'text': [
                                       "FAM is the new mobile app which combines events and all your social media needs",
                                       "Destiny, NY - FINAL HOURS!",
                                       "A graphic novel about two magical ladies in love."],
                                   'label': [3, 6, 4]})
            df_test = pd.DataFrame({'height': [2, 5, 3], 'key_words': ['hello', 'yes', 'yes'],
                                    'text': [
                                        "Publishing Magus Magazine,We are publishing a magazine that focuses on the folklore of the occult and paranormal.",
                                        "It is tabloid format but with academic articles",
                                        "a strong-willed Russian madam and The Cross at its most fabulous."],
                                    'label': [1, 4, 2]})
            if text_only:
                metadata = {'output_type': 'numbers',
                            'input_features': ['text'],
                            'output_label': ['label'],
                            'input_text': ['text'],
                            'input_bool': [],
                            'input_categorical': [],
                            'input_datetime': [],
                            'input_int': [],
                            'input_float': []
                            }

            else:
                metadata = {'output_type': 'numbers',
                            'input_features': ['height', 'key_words', 'text'],
                            'output_label': ['label'],
                            'input_text': ['text'],
                            'input_bool': [],
                            'input_categorical': ['key_words'],
                            'input_datetime': [],
                            'input_int': ['height'],
                            'input_float': []
                            }
        else:
            df_train = pd.DataFrame({'height': [1, 2, 3], 'key_words': ['hello', 'hi', 'yes'], 'label': [0, 1, 2]})
            df_dev = pd.DataFrame({'height': [4, 7, 5], 'key_words': ['hi', 'hi', 'yes'], 'label': [3, 6, 4]})
            df_test = pd.DataFrame({'height': [2, 5, 3], 'key_words': ['hello', 'yes', 'yes'], 'label': [1, 4, 2]})
            metadata = {'output_type': 'numbers',
                        'input_features': ['height', 'key_words'],
                        'output_label': ['label'],
                        'input_text': [],
                        'input_bool': [],
                        'input_categorical': ['key_words'],
                        'input_datetime': [],
                        'input_int': ['height'],
                        'input_float': []
                        }
    else:
        raise ValueError('Unknown task output_type: {}'.format(output_type))
    return df_train, df_dev, df_test, metadata

def get_fake_dataset_binary_class(with_text_col=False, text_only=False, output_type='classes'):
## you can change this to create your own test dataset here ##
    if with_text_col:
        df_train = pd.DataFrame({'height': [1,2,3], 'key_words': ['hello', 'hi', 'yes'],
                                 'text': ["Strange Wit, an original graphic novel about Jane Bowles",
                                          "The true biography of the historical figure, writer, alcoholic, lesbian",
                                          "world traveler: Jane Sydney Auer Bowles."],
                                 'label': [0, 1, 1]})
        df_dev = pd.DataFrame({'height': [4,7,5], 'key_words': ['hi', 'hi', 'yes'],
                               'text': ["FAM is the new mobile app which combines events and all your social media needs",
                                        "Destiny, NY - FINAL HOURS!",
                                        "A graphic novel about two magical ladies in love."],
                               'label': [1, 1, 0]})
        df_test = pd.DataFrame({'height': [2,5,3], 'key_words': ['hello', 'yes', 'yes'],
                                'text':["Publishing Magus Magazine,We are publishing a magazine that focuses on the folklore of the occult and paranormal.",
                                        "It is tabloid format but with academic articles",
                                        "a strong-willed Russian madam and The Cross at its most fabulous."],
                                'label': [0, 1, 0]})
        if text_only:
            metadata = {'output_type': 'classes',
                        'input_features': ['text'],
                        'output_label': ['label'],
                        'input_text': ['text'],
                        'input_bool': [],
                        'input_categorical': [],
                        'input_datetime': [],
                        'input_int': [],
                        'input_float': []
                        }

        else:
            metadata = {'output_type': 'classes',
                        'input_features': ['height','key_words','text'],
                        'output_label': ['label'],
                        'input_text': ['text'],
                        'input_bool': [],
                        'input_categorical': ['key_words'],
                        'input_datetime': [],
                        'input_int': ['height'],
                        'input_float': []
                        }
    else:
        # df_train = pd.DataFrame({'height': [1, 2, 1], 'key_words': ['hello', 'hi', 'hello'], 'label': [0, 1, 0]})
        # df_dev = pd.DataFrame({'height': [4, 7, 5], 'key_words': ['hi', 'hi', 'yes'], 'label': [1, 1, 0]})
        # df_test = pd.DataFrame({'height': [2, 5, 3], 'key_words': ['hello', 'yes', 'yes'], 'label': [0, 1, 0]})
        df_train = pd.DataFrame({'height': [0, 1, 0], 'label': [1, 0, 1]})
        df_dev = pd.DataFrame({'height': [4,7,5], 'label': [1, 1, 0]})
        df_test = pd.DataFrame({'height': [2,5,3], 'label': [0, 1, 0]})
        metadata = {'output_type': 'classes',
                    'input_features': ['height'],
                    'output_label': ['label'],
                    'input_text': [],
                    'input_bool': [],
                    'input_categorical': [],
                    'input_datetime': [],
                    'input_int': ['height'],
                    'input_float': []
                    }

    return df_train, df_dev, df_test, metadata

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
    
    # Create metadata with proper output structure
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

class TestEncoder(unittest.TestCase):
    def test_strucdata_only_numerical_outputs(self):
        df_train, df_dev, df_test, metadata = get_fake_dataset(with_text_col=False, output_type='numbers')
        encoder = Encoder(metadata, text_config=None)

        y_train, X_train, _ = encoder.fit_transform(df_train)
        y_dev, X_dev, _ = encoder.transform(df_dev)
        y_test, X_test, _ = encoder.transform(df_test)

        print('*' * 20)
        print(y_dev)
        print('*' * 20)

        X_train_true = np.array([
            [-1.22474487,  1.        ,  0.        ,  0.        ],
            [ 0.        ,  0.        ,  1.        ,  0.        ],
            [ 1.22474487,  0.        ,  0.        ,  1.        ]])
        y_train_true = np.array([
            [0],
            [1],
            [2]])
        self.assertTrue(np.isclose(X_train_true, X_train).all())
        self.assertTrue(np.isclose(y_train_true, y_train).all())
        X_dev_true = np.array([
            [2.44948974, 0., 1., 0.],
            [6.12372436, 0., 1., 0.],
            [3.67423461, 0., 0., 1.]])
        y_dev_true = np.array([
            [3],
            [6],
            [4]])
        self.assertTrue(np.isclose(X_dev_true, X_dev).all())
        self.assertTrue(np.isclose(y_dev_true, y_dev).all())
        X_test_true = np.array([
            [0., 1., 0., 0.],
            [3.67423461, 0., 0., 1.],
            [1.22474487, 0., 0., 1.]])
        y_test_true = np.array([
            [1],
            [4],
            [2]])
        self.assertTrue(np.isclose(X_test_true, X_test).all())
        self.assertTrue(np.isclose(y_test_true, y_test).all())

    def test_strucdata_only(self):
        df_train, df_dev, df_test, metadata = get_fake_dataset(with_text_col=False)
        encoder = Encoder(metadata, text_config=None)

        y_train, X_train, _ = encoder.fit_transform(df_train)
        y_dev, X_dev, _ = encoder.transform(df_dev)
        y_test, X_test, _ = encoder.transform(df_test)

        X_train_true = np.array([
            [-1.22474487,  1.        ,  0.        ,  0.        ], 
            [ 0.        ,  0.        ,  1.        ,  0.        ], 
            [ 1.22474487,  0.        ,  0.        ,  1.        ]])
        y_train_true = np.array([
            [0],
            [1],
            [2]])
        self.assertTrue(np.isclose(X_train_true, X_train).all())
        self.assertTrue(np.isclose(y_train_true, y_train).all())
        X_dev_true = np.array([
            [2.44948974, 0.        , 1.        , 0.        ],
            [6.12372436, 0.        , 1.        , 0.        ],
            [3.67423461, 0.        , 0.        , 1.        ]])
        y_dev_true = np.array([
            [1],
            [1],
            [2]])
        self.assertTrue(np.isclose(X_dev_true, X_dev).all())
        self.assertTrue(np.isclose(y_dev_true, y_dev).all())
        X_test_true = np.array([
            [0.        , 1.        , 0.        , 0.        ],
            [3.67423461, 0.        , 0.        , 1.        ],
            [1.22474487, 0.        , 0.        , 1.        ]])
        y_test_true = np.array([
            [2],
            [1],
            [2]])
        self.assertTrue(np.isclose(X_test_true, X_test).all())
        self.assertTrue(np.isclose(y_test_true, y_test).all())


    def test_tfidf(self):
        df_train, df_dev, df_test, metadata = get_fake_dataset(with_text_col=True)

        text_config = Mapping()
        text_config.mode = 'tfidf'
        text_config.max_words = 20
        print('*' * 20)
        print(text_config.mode)

        encoder = Encoder(metadata, text_config=text_config)
        y_train, X_train_struc, X_train_text = encoder.fit_transform(df_train)
        y_dev, X_dev_struc, X_dev_text = encoder.transform(df_dev)
        y_test, X_test_struc, X_test_text = encoder.transform(df_test)

        X_train_text_true = np.array([
            [0.        , 0.69314718, 0.69314718, 0.        , 0.91629073,
            0.91629073, 0.91629073, 0.91629073, 0.91629073, 0.91629073,
            0.91629073, 0.        , 0.        , 0.        , 0.        ,
            0.        , 0.        , 0.        , 0.        , 0.        ],
           [0.        , 0.        , 0.        , 1.55141507, 0.        ,
            0.        , 0.        , 0.        , 0.        , 0.        ,
            0.        , 0.91629073, 0.91629073, 0.91629073, 0.91629073,
            0.91629073, 0.91629073, 0.91629073, 0.91629073, 0.        ],
           [0.        , 0.69314718, 0.69314718, 0.        , 0.        ,
            0.        , 0.        , 0.        , 0.        , 0.        ,
            0.        , 0.        , 0.        , 0.        , 0.        ,
            0.        , 0.        , 0.        , 0.        , 0.91629073]])
        X_train_struc_true = np.array([
            [-1.22474487,  1.        ,  0.        ,  0.        ],
            [0.        ,  0.        ,  1.        ,  0.        ],
            [1.22474487,  0.        ,  0.        ,  1.        ]])
        self.assertTrue(np.isclose(X_train_text_true, X_train_text).all())
        self.assertTrue(np.isclose(X_train_struc_true, X_train_struc).all())
        X_dev_text_true = np.array([
            [0.        , 0.        , 0.        , 0.91629073, 0.        ,
            0.        , 0.        , 0.        , 0.        , 0.        ,
            0.        , 0.        , 0.        , 0.        , 0.        ,
            0.        , 0.        , 0.        , 0.        , 0.        ],
           [0.        , 0.        , 0.        , 0.        , 0.        ,
            0.        , 0.        , 0.        , 0.        , 0.        ,
            0.        , 0.        , 0.        , 0.        , 0.        ,
            0.        , 0.        , 0.        , 0.        , 0.        ],
           [0.        , 0.        , 0.        , 0.        , 0.        ,
            0.        , 0.        , 0.        , 0.91629073, 0.91629073,
            0.91629073, 0.        , 0.        , 0.        , 0.        ,
            0.        , 0.        , 0.        , 0.        , 0.        ]])
        X_dev_struc_true = np.array([
            [2.44948974, 0.        , 1.        , 0.        ],
            [6.12372436, 0.        , 1.        , 0.        ],
            [3.67423461, 0.        , 0.        , 1.        ]])
        self.assertTrue(np.isclose(X_dev_text_true, X_dev_text).all())
        self.assertTrue(np.isclose(X_dev_struc_true, X_dev_struc).all())

        X_test_text_true = np.array([
            [0.        , 0.        , 0.        , 1.55141507, 0.        ,
            0.        , 0.        , 0.        , 0.        , 0.        ,
            0.        , 0.        , 0.        , 0.91629073, 0.        ,
            0.        , 0.        , 0.        , 0.        , 0.        ],
           [0.        , 0.        , 0.        , 0.        , 0.        ,
            0.        , 0.        , 0.        , 0.        , 0.        ,
            0.        , 0.        , 0.        , 0.        , 0.        ,
            0.        , 0.        , 0.        , 0.        , 0.        ],
           [0.        , 0.        , 0.        , 0.91629073, 0.        ,
            0.        , 0.        , 0.        , 0.        , 0.        ,
            0.        , 0.        , 0.        , 0.        , 0.        ,
            0.        , 0.        , 0.        , 0.        , 0.        ]])
        X_test_struc_true = np.array([
            [0.        , 1.        , 0.        , 0.        ],
            [3.67423461, 0.        , 0.        , 1.        ],
            [1.22474487, 0.        , 0.        , 1.        ]])
        self.assertTrue(np.isclose(X_test_text_true, X_test_text).all())
        self.assertTrue(np.isclose(X_test_struc_true, X_test_struc).all())


    def test_word_embedding(self):
        df_train, df_dev, df_test, metadata = get_fake_dataset(with_text_col=True)

        glove_file_path = 'resource/glove/glove.6B.50d.txt'# need be changed to where you store the pre-trained GloVe file.
        
        text_config = Mapping()
        text_config.mode = 'glove'
        text_config.max_words = 20
        text_config.maxlen = 5
        text_config.embedding_dim = 50
        text_config.embeddings_index = open_glove(glove_file_path)

        encoder = Encoder(metadata, text_config=text_config)
        y_train, X_train_struc, X_train_text = encoder.fit_transform(df_train)
        y_dev, X_dev_struc, X_dev_text = encoder.transform(df_dev)
        y_test, X_test_struc, X_test_text = encoder.transform(df_test)

        X_train_text_true = np.array([
            [ 9, 10, 11,  2,  3],
            [15, 16, 17, 18, 19],
            [1,  2, 1, 1,  3]])
        X_train_struc_true = np.array([
            [-1.22474487,  1.        ,  0.        ,  0.        ],
            [ 0.        ,  0.        ,  1.        ,  0.        ],
            [ 1.22474487,  0.        ,  0.        ,  1.        ]])
        self.assertTrue(np.isclose(X_train_text_true, X_train_text).all())
        self.assertTrue(np.isclose(X_train_struc_true, X_train_struc).all())
        X_dev_text_true = np.array([
            [1, 1, 1, 1, 1],
           [1, 1, 1, 1,  0],
           [1, 1, 1, 1, 1]])
        X_dev_struc_true = np.array([
            [2.44948974, 0.        , 1.        , 0.        ],
            [6.12372436, 0.        , 1.        , 0.        ],
            [3.67423461, 0.        , 0.        , 1.        ]])
        self.assertTrue(np.isclose(X_dev_text_true, X_dev_text).all())
        self.assertTrue(np.isclose(X_dev_struc_true, X_dev_struc).all())
        X_test_text_true = np.array([
            [14,  4, 1, 1, 1],
           [1, 1, 1, 1, 1],
           [1, 1, 1, 1, 1]])
        X_test_struc_true = np.array([
            [0.        , 1.        , 0.        , 0.        ],
            [3.67423461, 0.        , 0.        , 1.        ],
            [1.22474487, 0.        , 0.        , 1.        ]])
        self.assertTrue(np.isclose(X_test_text_true, X_test_text).all())
        self.assertTrue(np.isclose(X_test_struc_true, X_test_struc).all())

    def test_encoder_multi_task(self):
        """Test encoder with multiple tasks (classification and regression)"""
        # Generate fake dataset with multiple targets
        df_train, df_dev, df_test, metadata = get_fake_dataset_multi_task(
            with_text_col=True,
            classification_targets=['sentiment'],  # Binary classification
            regression_targets=['rating']          # Regression
        )
        
        # Add required metadata fields for multi-task
        metadata.update({
            'output_type': 'multi_task',
            'task_types': {
                'sentiment': 'classification',
                'rating': 'regression'
            }
        })
        
        # Configure text processing
        text_config = Mapping()
        text_config.mode = 'tfidf'
        text_config.max_words = 20
        
        # Create encoder and test
        encoder = Encoder(metadata, text_config=text_config)
        y_train_dict, X_train_struc, X_train_text = encoder.fit_transform_multi_task(df_train)
        
        # Check outputs structure
        self.assertIn('sentiment_output', y_train_dict)
        self.assertIn('rating_output', y_train_dict)
        self.assertIsNotNone(X_train_struc)
        self.assertIsNotNone(X_train_text)
        
        # Check shapes
        self.assertEqual(y_train_dict['sentiment_output'].shape[0], len(df_train))
        self.assertEqual(y_train_dict['rating_output'].shape[0], len(df_train))
        self.assertEqual(X_train_struc.shape[0], len(df_train))
        self.assertEqual(X_train_text.shape[0], len(df_train))
        
        # Test transform_multi_task
        y_dev_dict, X_dev_struc, X_dev_text = encoder.transform_multi_task(df_dev)
        
        # Check dev outputs
        self.assertIn('sentiment_output', y_dev_dict)
        self.assertIn('rating_output', y_dev_dict)
        self.assertEqual(y_dev_dict['sentiment_output'].shape[0], len(df_dev))
        self.assertEqual(y_dev_dict['rating_output'].shape[0], len(df_dev))
        self.assertEqual(X_dev_struc.shape[0], len(df_dev))
        self.assertEqual(X_dev_text.shape[0], len(df_dev))
        
        # Check feature dimensions consistency
        self.assertEqual(X_train_struc.shape[1], X_dev_struc.shape[1])
        self.assertEqual(X_train_text.shape[1], X_dev_text.shape[1])

    def test_encoder_multi_task_classification_only(self):
        """Test encoder with multiple classification tasks"""
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
        
        # Create encoder
        encoder = Encoder(metadata, text_config=text_config)
        
        # Test fit_transform_multi_task
        y_train_dict, X_train_struc, X_train_text = encoder.fit_transform_multi_task(df_train)
        
        # Check outputs structure
        self.assertIn('sentiment_output', y_train_dict)
        self.assertIn('topic_output', y_train_dict)
        
        # Check binary vs multi-class shapes
        sentiment_values = np.unique(y_train_dict['sentiment_output'])
        topic_values = np.unique(y_train_dict['topic_output'])
        self.assertEqual(len(sentiment_values), 2)  # Binary
        self.assertEqual(len(topic_values), 3)      # Multi-class
        
        # Test transform_multi_task
        y_dev_dict, X_dev_struc, X_dev_text = encoder.transform_multi_task(df_dev)
        
        # Check consistency
        self.assertEqual(set(y_train_dict.keys()), set(y_dev_dict.keys()))
        self.assertEqual(X_train_struc.shape[1], X_dev_struc.shape[1])
        self.assertEqual(X_train_text.shape[1], X_dev_text.shape[1])

if __name__ == '__main__':
    if not os.path.exists('tmp'):
        os.makedirs('tmp')    
    unittest.main()
