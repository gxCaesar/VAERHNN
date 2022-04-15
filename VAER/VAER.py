
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
    X, y,test_size=validation_size, random_state=seed)

# Evaluation Algorithm-Evaluation Criteria
num_folds = 10
scoring = 'neg_mean_squared_error'

#models definitely
reg_AB = AdaBoostRegressor(n_estimators=148)
reg_XG = XGBRegressor(n_estimators=22,learning_rate=0.3,max_depth=11,gamma=0.01)
reg_GBM = GradientBoostingRegressor(n_estimators=148,learning_rate=0.1,max_depth=1,alpha=0.083)
reg_HGBM = HistGradientBoostingRegressor(learning_rate=0.06,max_depth=2,max_iter=60)
reg_LGBM = LGBMRegressor(colsample_bytree=0.4, learning_rate=0.08, max_depth=1, n_estimators=200, num_leaves=2, subsample=1)
reg_Cat = CatBoostRegressor(n_estimators=370,learning_rate=0.1,depth=5,border_count=10)
reg_Bag = BaggingRegressor(n_estimators=187)
reg_RF = RandomForestRegressor(n_estimators=142,max_depth=17)
reg_ET = ExtraTreesRegressor(n_estimators=15,criterion='mse', max_depth=19)
AB_ET  = AdaBoostRegressor(base_estimator=reg_ET,n_estimators=148)

#Optimal algorithm after tuning
pipelines_AB = Pipeline([('Scaler', StandardScaler()), ('AB',AdaBoostRegressor(n_estimators=148))])
pipelines_XG = Pipeline([('Scaler', StandardScaler()), ('XG',XGBRegressor(n_estimators=22,learning_rate=0.3,max_depth=11,gamma=0.01))])
pipelines_GBM = Pipeline([('Scaler', StandardScaler()), ('GBM',GradientBoostingRegressor(n_estimators=148,learning_rate=0.1,max_depth=1,alpha=0.083))])
pipelines_HGBM = Pipeline([('Scaler', StandardScaler()), ('HGBM',LGBMRegressor(colsample_bytree=0.4, learning_rate=0.08, max_depth=1, n_estimators=200, num_leaves=2, subsample=1))])
pipelines_LGBM = Pipeline([('Scaler', StandardScaler()), ('LGBM',LGBMRegressor(colsample_bytree=0.4, learning_rate=0.08, max_depth=1, n_estimators=200, num_leaves=2, subsample=1))])
pipelines_Cat = Pipeline([('Scaler', StandardScaler()), ('Cat',CatBoostRegressor(n_estimators=370,learning_rate=0.1,depth=5,border_count=10))])
pipelines_Bag = Pipeline([('Scaler', StandardScaler()), ('Bag',BaggingRegressor(n_estimators=187))])
pipelines_RF = Pipeline([('Scaler', StandardScaler()), ('RF',RandomForestRegressor(n_estimators=142,max_depth=17))])
pipelines_ET = Pipeline([('Scaler', StandardScaler()), ('ET',ExtraTreesRegressor(n_estimators=15,criterion='mse', max_depth=19))])
pipelines_AB_ET = Pipeline([('Scaler', StandardScaler()), ('AB_ET',AdaBoostRegressor(base_estimator=reg_ET,n_estimators=148))])

# combine algorithm
estimators = [('AB',pipelines_AB)
              ,('XG',pipelines_XG)
              ,('GBM',pipelines_GBM)
              ,('HGBM',pipelines_HGBM)
              ,('LGBM',pipelines_LGBM)
              ,('Cat',pipelines_Cat)
              ,('Bagging',pipelines_Bag)
              ,('RF',pipelines_RF)
              ,('ET',pipelines_ET)
              ,('AB-ET',pipelines_AB_ET)
             ]
stacking_regressor = StackingRegressor(estimators = estimators)
Voting_regressor = VotingRegressor(estimators = estimators)


#Measure and plot the results of Stacking
sns.set_style("white")
def plot_regression_results(ax, y_true, y_pred, title, scores, elapsed_time):
    """Scatter plot of the predicted vs true targets."""
    ax.plot([y_true.min(), y_true.max()],
            [y_true.min(), y_true.max()],
            '--',color='#900302', linewidth=2,alpha=0.2)
    ax.scatter(y_true, y_pred, alpha=0.2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    ax.spines['left'].set_position(('outward', 10))
    ax.spines['bottom'].set_position(('outward', 10))
    ax.set_xlim([y_true.min(), y_true.max()])
    ax.set_ylim([y_true.min(), y_true.max()])
    ax.set_xlabel('Actual pIC50')
    ax.set_ylabel('Predicted pIC50')
    extra = plt.Rectangle((0, 0), 0, 0, fc="w", fill=False,
                          edgecolor='none', linewidth=0)
    ax.legend([extra], [scores], loc='upper left')
    title =  '\n'+ title
    ax.set_title(title,size = 12)

fig, axs = plt.subplots(4,3 , figsize=(12,15))
axs = np.ravel(axs)
for ax, (name, est) in zip(axs, estimators + [('Stacking',stacking_regressor)]+[('Voting',Voting_regressor)]):
    start_time = time.time()
    score = cross_validate(est, X, y,
                           scoring=['neg_mean_squared_error','neg_mean_absolute_error'],
                           n_jobs=-1, verbose=0)
    elapsed_time = time.time() - start_time
    y_pred = cross_val_predict(est, X, y, n_jobs=-1, verbose=0)

    plot_regression_results(ax, y, y_pred,name,(r'$MSE={:.3f} \pm {:.3f}$' + '\n' + r'$MAE={:.3f} \pm {:.3f}$').format(-np.mean(score['test_neg_mean_squared_error']),np.std(score['test_neg_mean_squared_error']),-np.mean(score['test_neg_mean_absolute_error']),np.std(score['test_neg_mean_absolute_error'])),elapsed_time)

plt.suptitle('\n'+'\n'+'The performance of single predictors versus stacked predictors',size=16)
plt.tight_layout()
plt.subplots_adjust(top=0.9)
#plt.savefig('image/3_StandardScalerRegression.png',dpi=600,bbox_inches='tight')
plt.show()

zinc = pd.read_csv('data/candidate.csv')
array = zinc.values
data_pred = array[:, 2:]

pipelines_AB.fit(X_train,y_train)
pipelines_XG.fit(X_train,y_train)
pipelines_GBM.fit(X_train,y_train)
pipelines_HGBM.fit(X_train,y_train)
pipelines_LGBM.fit(X_train,y_train)
pipelines_Cat.fit(X_train,y_train)
pipelines_Bag.fit(X_train,y_train)
pipelines_RF.fit(X_train,y_train)
pipelines_ET.fit(X_train,y_train)
pipelines_AB_ET.fit(X_train,y_train)
stacking_regressor.fit(X_train,y_train)

ereg = VotingRegressor([('AB', pipelines_AB), ('XG', reg_XG), ('GBM',pipelines_GBM),('HGBM',reg_HGBM),('LGBM',reg_LGBM),('Cat',pipelines_Bag),('Bagging',pipelines_Bag),('RF',pipelines_RF),('ET', pipelines_ET),('AB-ET',pipelines_AB_ET),('Stack',stacking_regressor)])
ereg.fit(X_train,y_train)

pred1 = pipelines_AB.predict(data_pred)
pred2 = pipelines_XG.predict(data_pred)
pred3 = pipelines_GBM.predict(data_pred)
pred4 = pipelines_HGBM.predict(data_pred)
pred5 = pipelines_LGBM.predict(data_pred)
pred6 = pipelines_Cat.predict(data_pred)
pred7 = pipelines_Bag.predict(data_pred)
pred8 = pipelines_RF.predict(data_pred)
pred9 = pipelines_ET.predict(data_pred)
pred10 = pipelines_AB_ET.predict(data_pred)
pred11 = stacking_regressor.predict(data_pred)
pred12 = ereg.predict(data_pred)
#result_pred = np.vstack((pred1,pred2,pred3,pred4,pred5,pred6,pred7,pred8,pred9,pred10,pred11,pred12))
#np.savetxt('result/results_zinc.csv',result_pred,delimiter=',')


# In[46]:


sns.set_style('darkgrid')
sns.set_palette('deep')
f, ax = plt.subplots(figsize=(10,5))
plt.plot(pred1,'v',label='ET')
plt.plot(pred2, 'x',label='XG')
plt.plot(pred3, 's',label='AB')
plt.plot(pred4, 'p',label='GBM')
plt.plot(pred5, 'h',label='HGBM')
plt.plot(pred6, 'd',label='LGBM')
plt.plot(pred7, '1',label='CatB')
plt.plot(pred8, 'o',label='Bagging')
plt.plot(pred9, '2',label='RF')
plt.plot(pred10, '^',label='AB-ET')
plt.plot(pred11, '>',label='Stack')
plt.plot(pred12,'*', ms=10, color='#900302',label='Voting')
plt.ylabel('Predicted PIC50',size=15)
#plt.xlabel('Candidates and control',size=12)
plt.xticks([0,1,2,3,4,5,6,7,8,9,10],['1','2', '3' ,'4', '5', '6', '7', '8' ,'9', '10', 'control'],rotation=45, size=15)
plt.legend(bbox_to_anchor=(1.01, 0), loc=3, borderaxespad=0,fontsize=13)
plt.title('Predictions of regression estimators and VAER',size = 15)
plt.tight_layout() 
#plt.savefig('image/4_VotingPredictors_zinc.png',dpi=600,bbox_inches='tight')
plt.show()

