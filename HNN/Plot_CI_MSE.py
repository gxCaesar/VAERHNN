#!/usr/bin/env python
# coding: utf-8

import seaborn as sns
import numpy as np
import matplotlib as plt
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pandas as pd


data = pd.read_csv('result/mse.csv')
sns.set_style("ticks")
f,ax = plt.subplots(figsize=(10,4)) 
sns.heatmap(data, annot=True, linewidths=.5, fmt='.3f',cmap='Pastel1', yticklabels =['AAC','PseAAC','ConTriad','Quasi-seq','CNN','GRU','LSTM','Transformer'])
ax.set_xlabel('Drug Model',size = 13)
ax.set_ylabel('Target Model',size = 13)
ax.set_title('The MSE of Multi-NN model',size=15)
#plt.savefig('MSE.png',dpi=600,bbox_inches="tight")
plt.show()

data2 = pd.read_csv('result/CI.csv')
sns.set_style("ticks")
f,ax = plt.subplots(figsize=(10,4)) 
sns.heatmap(data2, annot=True, linewidths=.5, fmt='.3f',cmap='Pastel1_r', yticklabels =['AAC','PseAAC','ConTriad','Quasi-seq','CNN','GRU','LSTM','Transformer'])
ax.set_xlabel('Drug Model',size = 13)
ax.set_ylabel('Target Model',size = 13)
ax.set_title('The C-index of Multi-NN model',size=15)
#plt.savefig('CI.png',dpi=600,bbox_inches="tight")
plt.show()