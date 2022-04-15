#!/usr/bin/env python
# coding: utf-8

# Import library
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
from numpy import arange
import matplotlib.pyplot as plt
import seaborn as sns


# Computing weights
CI= pd.read_csv('result/CI_weight.csv').iloc[:,1]
weights = CI.apply(lambda x : x/CI.sum())

# Individual predictions
dta = pd.read_csv('result/HNN.csv').iloc[:,1:]
pred1 = dta.iloc[:,0] 
pred2 = dta.iloc[:,1] 
pred3 = dta.iloc[:,2] 
pred4 = dta.iloc[:,3] 
pred5 = dta.iloc[:,4] 
pred6 = dta.iloc[:,5] 
pred7 = dta.iloc[:,6] 
pred8 = dta.iloc[:,7] 
pred9 = dta.iloc[:,8]

# Computing the voting-averaged predictions by HNN
va1 = np.average(dta.loc[0], weights=weights)
va2 = np.average(dta.loc[1], weights=weights)
va3 = np.average(dta.loc[2], weights=weights)
va4 = np.average(dta.loc[3], weights=weights)
va5 = np.average(dta.loc[4], weights=weights)
va6 = np.average(dta.loc[5], weights=weights)
va7 = np.average(dta.loc[6], weights=weights)
va8 = np.average(dta.loc[7], weights=weights)
va9 = np.average(dta.loc[8], weights=weights)
va10 = np.average(dta.loc[9], weights=weights)
va11 = np.average(dta.loc[10], weights=weights)
hnn = [va1,va2,va3,va4,va5,va6,va7,va8,va9,va10,va11]

# Plot 
sns.set_style('darkgrid')
sns.set_palette('deep')
f, ax = plt.subplots(figsize=(10,5))
plt.plot(pred1,'v',label='Morgan-AAC',alpha=0.5)
plt.plot(pred2, 'x',label='Morgan-Quasi-seq',alpha=0.5)
plt.plot(pred3, 's',label='Morgan-CNN',alpha=0.5)
plt.plot(pred4, 'p',label='Morgan-GRU',alpha=0.5)
plt.plot(pred5, 'h',label='Morgan-LSTM',alpha=0.5)
plt.plot(pred6, 'd',label='Morgan-Transformer',alpha=0.5)
plt.plot(pred7, '1',label='PubChem-AAC',alpha=0.5)
plt.plot(pred8, 'o',label='PubChem-Transformer',alpha=0.5)
plt.plot(pred9, '2',label='RDKit-2D-Transformer',alpha=0.5)
plt.plot(hnn,'*', ms=10, color='#900302',label='HNN')
plt.ylabel('Final score',size=15)
plt.yticks(size=15)
# plt.xlabel('Candidates from virtual screening and Terazosin',size=15)
# plt.xticks([0,1,2,3,4,5,6,7,8,9,10],['ZINC8215434','ZINC95617639', 'ZINC8215403' ,'ZINC3914596', 'ZINC14768621', 'ZINC11616925', 'ZINC4215255', 'ZINC2036915' ,'ZINC1536109', 'ZINC95617641', 'Terazosin'],rotation=45, size=15)
plt.legend(bbox_to_anchor=(1.01, 0), loc=3, borderaxespad=0,fontsize=13)
plt.tight_layout() 
# plt.savefig('image/HNN.png',dpi=600,bbox_inches='tight')
plt.show()

