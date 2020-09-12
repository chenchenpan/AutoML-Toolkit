#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import json
import pickle as pkl
import numpy as np
import pandas as pd
import sklearn.metrics

from analysis_util import Encoder, Mapping, read_file, load_all, get_best_trial

def shuffle_col(df, col, seed=None):
    new_df = df.copy()
    if seed is None: 
        new_df[col] = np.random.permutation(new_df[col])
    else:
        np.random.seed(seed)
        new_df[col] = np.random.permutation(new_df[col])
    return new_df

def evaluate(df, encoder, model):
    y, X_struc, X_text = encoder.transform(df)
    print(y.shape)
    # print(X_struc.shape)
    # print(X_text.shape)
    preds = model.predict(X_struc, X_text)
    if len(y.shape) > 1 and y.shape[-1] > 1:
        y = np.argmax(y, axis=-1)
    print(y.shape)
    print(preds.shape)
    acc = sklearn.metrics.accuracy_score(y, preds)
    return acc


# In[62]:

output_dir = '/datadrive/projects/AutoML/NPS/outputs/nn_outputs/'
data_file = '/datadrive/projects/AutoML/NPS/data/raw_data/NPS_dev_clean.tsv'
label_col = 'NPS'

# output_dir = '/datadrive/projects/AutoML/demo/outputs/nn_outputs/'
# data_file = '/datadrive/projects/AutoML/demo/data/raw_data/comb_dev.tsv'
# label_col = 'label'

df = read_file(data_file)


# In[63]:


best_trial = get_best_trial(output_dir)
model, encoder = load_all(best_trial)


# In[64]:


# encoder.text_config.mode


# In[65]:


cols = list(df.columns)
cols


# In[66]:


df.head()


# In[67]:


# shuffle_col(df, 'desc_clean').head()


# In[68]:


original_metrics = evaluate(df, encoder, model)
original_metrics


# In[69]:


original_metrics = evaluate(df, encoder, model)

n_samples = 10

feature_importance_dict = {}

for col in cols:
    if col == label_col:
        continue
    new_metrics = []
    for i in range(n_samples):
        new_df = shuffle_col(df, col, seed=i)
        metric = evaluate(new_df, encoder, model)
        new_metrics.append(metric)
    mean = np.mean(new_metrics)
    feature_importance_dict[col] = original_metrics - mean


# In[70]:


for col, importance in sorted(feature_importance_dict.items(), key=lambda x: x[-1], reverse=True):
    print('{}: {}'.format(col, importance))

