import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import seaborn as sns
import gc
from sklearn.exceptions import NotFittedError
from itertools import chain
from  feature_selector import FeatureSelector
import os

train = pd.read_csv('data/GFA_features.csv')
train = train.sample(10000, replace=True)
train_labels = train['pIC50']
train = train.drop(columns = 'pIC50')
fs = FeatureSelector(data = train, labels = train_labels)

#1.Missing Values
fs.identify_missing(missing_threshold=0.6)
missing_features = fs.ops['missing']
missing_features[:10]
fs.plot_missing()
# plt.savefig('images/missing.png',dpi=600,bbox_inches="tight")

#2. Single Unique Value
fs.identify_single_unique()
single_unique = fs.ops['single_unique']
single_unique
fs.plot_unique()
# plt.savefig('images/uniquq.png',dpi=600,bbox_inches="tight")

#3. Collinear (highly correlated) Features
fs.identify_collinear(correlation_threshold=0.975)
correlated_features = fs.ops['collinear']
correlated_features[:5]
fs.plot_collinear()
# plt.savefig('images/collinear0.975.png')
fs.plot_collinear(plot_all=True)
plt.savefig('images/allcollinear.png',dpi=600,bbox_inches="tight")
fs.identify_collinear(correlation_threshold=0.99)
fs.plot_collinear()
#plt.savefig('images/collinear0.99.png',dpi=600,bbox_inches="tight")

#4. Zero Importance Features
fs.identify_zero_importance(task = 'regression', eval_metric = 'l2',
                            n_iterations = 10, early_stopping = True)
one_hot_features = fs.one_hot_features
base_features = fs.base_features
print('There are %d original features' % len(base_features))
print('There are %d one-hot features' % len(one_hot_features))

zero_importance_features = fs.ops['zero_importance']
zero_importance_features[10:15]

#Plot Feature Importances
fs.plot_feature_importances(threshold = 0.99, plot_n = 12)
one_hundred_features = list(fs.feature_importances.loc[:99, 'feature'])
len(one_hundred_features)
# plt.savefig('images/featureImportance.png',dpi=600,bbox_inches="tight")

#5.Low Importance Features
fs.identify_low_importance(cumulative_importance = 0.99)
low_importance_features = fs.ops['low_importance']
low_importance_features[:5]

#Removing Features
train_no_missing = fs.remove(methods = ['missing'])
train_no_missing_zero = fs.remove(methods = ['missing', 'zero_importance'])
all_to_remove = fs.check_removal()
all_to_remove[10:25]
train_removed = fs.remove(methods = 'all')

#Handling One-Hot Features
train_removed_all = fs.remove(methods = 'all', keep_one_hot=False)
print('Original Number of Features', train.shape[1])
print('Final Number of Features: ', train_removed_all.shape[1])

#Alternative Option for Using all Methods
fs = FeatureSelector(data = train, labels = train_labels)

fs.identify_all(selection_params = {'missing_threshold': 0.6, 'correlation_threshold': 0.98,
                                    'task': 'regression', 'eval_metric': 'l2',
                                     'cumulative_importance': 0.99})
train_removed_all_once = fs.remove(methods = 'all', keep_one_hot = True)
frame = train_removed_all_once.drop_duplicates()
frame.to_csv('result/selected_features.csv',encoding='utf8')