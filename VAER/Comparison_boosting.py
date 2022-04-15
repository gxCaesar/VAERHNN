
# coding: utf-8

# Import library
import warnings
warnings.filterwarnings('ignore')
import time
import pandas as pd
import numpy as np
from numpy import arange
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate,cross_val_predict
from sklearn.model_selection import GridSearchCV
#Import library of regression
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import GammaRegressor
from sklearn.linear_model import HuberRegressor
from sklearn.linear_model import PoissonRegressor
from sklearn.linear_model import TweedieRegressor
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.cross_decomposition import PLSRegression
#Import library of EL regression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from xgboost import XGBRegressor
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.ensemble import VotingRegressor
from sklearn.metrics import mean_squared_error


# Import Data
dataset  = pd.read_csv('data/inhibitor.csv')
# Separate the data set
array = dataset.values
X = array[:, 2:]
y = array[:, 1]
validation_size = 0.2
seed = 7
X_train, X_validation, y_train, y_validation = train_test_split(
    X, y,test_size=validation_size, random_state=seed, shuffle=True)

# Evaluation Algorithm-Evaluation Criteria
num_folds = 10
scoring = 'neg_mean_squared_error'

# In[2]:
# Evaluation Algorithm- Baseline
models = {}
models['Dummy'] = DummyRegressor()
models['Gamma'] = GammaRegressor()
models['Huber'] = HuberRegressor()
models['PR'] = PoissonRegressor()
models['TR'] = TweedieRegressor()
models['LASSO'] = Lasso()
models['EN'] = ElasticNet()
models['Ridge'] = Ridge()
models['KNN'] = KNeighborsRegressor()
models['SVM'] = SVR()
models['PLS'] = PLSRegression()

models['CART'] = DecisionTreeRegressor()
models['AB'] = AdaBoostRegressor()
models['XG'] = XGBRegressor()
models['GBM'] = GradientBoostingRegressor()
models['HGBM'] = HistGradientBoostingRegressor()
models['LBGM'] = LGBMRegressor()
models['Cat'] = CatBoostRegressor()
models['Bagging'] = BaggingRegressor()
models['RF'] = RandomForestRegressor()
models['ET'] = ExtraTreesRegressor()

# Evaluation Algorithm
results = []
for key in models:
    kfold = KFold(n_splits=num_folds, random_state=seed, shuffle=True)
    cv_result = cross_val_score(models[key], X_train, y_train, cv=kfold, scoring=scoring)
    results.append(cv_result)
    print('%s: %f (%f)' % (key, cv_result.mean(), cv_result.std()))
# np.savetxt('result/results_regAll.csv',results,delimiter=',')


# In[3]:
#Evaluation Algorithm-violinplot
sns.set_style("dark")
sns.set_context("paper")
f, ax = plt.subplots(figsize=(25,10))
ax.set_title('Performance comparison of 21 regression models',size=25)
plt.ylabel('Negative Mean Square Error(-MSE)',size=15)
sns.violinplot(data=results,whis="range",palette='deep',orient="v")
sns.swarmplot(data=results,size=2, color=".3", linewidth=0,orient="v")
ax.yaxis.grid(True)
ax.set_xticklabels(models.keys(),size=13)
#plt.savefig('image/1_AlgorithmComparison.png',dpi=600,whiskerprops=2,bbox_inches='tight')
plt.show()

# In[35]:


# Evaluation Algorithm- EL(ensemble learning)
models_EL = {}
models_EL['AB'] = AdaBoostRegressor(n_estimators=148)
models_EL['XG'] = XGBRegressor(n_estimators=22,learning_rate=0.3,max_depth=11,gamma=0.01)
models_EL['GBM'] = GradientBoostingRegressor(n_estimators=148,learning_rate=0.1,max_depth=1,alpha=0.083)
models_EL['HGBM'] = HistGradientBoostingRegressor(learning_rate=0.06,max_depth=2,max_iter=60)
models_EL['LGBM'] = LGBMRegressor(colsample_bytree=0.4, learning_rate=0.08, max_depth=1, n_estimators=200, num_leaves=2, subsample=1)
models_EL['CatB'] = CatBoostRegressor(n_estimators=370,learning_rate=0.1,depth=5,border_count=10)
models_EL['Bagging'] = BaggingRegressor(n_estimators=187)
models_EL['RF'] = RandomForestRegressor(n_estimators=142,max_depth=17)
models_EL['ET'] = ExtraTreesRegressor(n_estimators=15,criterion='mse', max_depth=19)
results_EL = []
for key in models_EL:
    kfold = KFold(n_splits=num_folds, random_state=seed, shuffle=True)
    cv_result = cross_val_score(models_EL[key], X_train, y_train, cv=kfold, scoring=scoring)
    results_EL.append(cv_result)
    print('%s: %f (%f)' % (key, cv_result.mean(), cv_result.std()))
#np.savetxt('result/results_EL.csv',results_EL,delimiter=',')


# In[ ]:


# Evaluation Algorithm- AB(Adaboost-X)
models_AB = {}
models_AB['AB-XG'] = AdaBoostRegressor(base_estimator=models_EL['XG'])
models_AB['AB-GBM'] = AdaBoostRegressor(base_estimator=models_EL['GBM'])
models_AB['AB-HGBM'] = AdaBoostRegressor(base_estimator=models_EL['HGBM'])
models_AB['AB-LGBM'] = AdaBoostRegressor(base_estimator=models_EL['LGBM'])
models_AB['AB-CatB'] = AdaBoostRegressor(base_estimator=models_EL['CatB'])
models_AB['AB-Bagging'] = AdaBoostRegressor(base_estimator=models_EL['Bagging'])
models_AB['AB-RF'] = AdaBoostRegressor(base_estimator=models_EL['RF'])
models_AB['AB-ET'] = AdaBoostRegressor(base_estimator=models_EL['ET'])
results_AB = []
for key in models_AB:
    kfold = KFold(n_splits=num_folds, random_state=seed, shuffle=True)
    cv_result = cross_val_score(models_AB[key], X_train, y_train, cv=kfold, scoring=scoring)
    results_AB.append(cv_result)
    print('%s: %f (%f)' % (key, cv_result.mean(), cv_result.std()))
#np.savetxt('result/results_AB.csv',results_AB,delimiter=',')


# In[4]:


#Evaluation Algorithm-violinplot
sns.set_style("dark")
sns.set_context("paper")
f, ax = plt.subplots(figsize=(20,8))
ax.set_title('Performance comparison of EL regression models with Adaboost',size=20)
plt.ylabel('Negative Mean Square Error(-MSE)',size=13)
sns.violinplot(data=results_AB_X,whis="range",palette='deep',orient="v")
sns.swarmplot(data=results_AB_X,size=2, color=".3", linewidth=0,orient="v")
ax.yaxis.grid(True)
#plt.savefig('image/2_results_AB_X.png',dpi=600,whiskerprops=2,bbox_inches='tight')
plt.show()





