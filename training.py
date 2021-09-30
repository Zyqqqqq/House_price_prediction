import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor
import lightgbm as lgb
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_boston
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

def load_data(path):
    data=pd.read_csv(path,encoding='gb18030')
    # data.drop(['ID'],axis='columns',inplace=True)
    return data

# 划分训练集、测试集
def splitData(data):
    train_set, test_set = train_test_split(data, test_size=0.2, random_state=42)
    y_train=np.log1p(train_set.pop('ESTA_TRAN_PRICE_UNIT'))
    X_train=train_set.values
    y_test=np.log1p(test_set.pop('ESTA_TRAN_PRICE_UNIT'))
    X_test=test_set.values
    return X_train,y_train,X_test,y_test

# 交叉验证岭回归
def ridge_train(data):
    X_train,y_train,X_test,y_test=splitData(data)
    test_scores=[]
    alphas=np.logspace(-3,2,50)
    for alpha in alphas:
        clf=Ridge(alpha)
        test_score=np.sqrt(-cross_val_score(clf,X_train,y_train,cv=10,scoring='neg_mean_squared_error'))
        test_scores.append(np.mean(test_score))
    plt.plot(alphas,test_scores)
    plt.title('Alpha vs cv error')
    plt.xlabel('Alpha')
    plt.ylabel('neg_mean_squared_error')
    plt.show()

# 交叉验证随机森林
def rf_train(data):
    X_train,y_train,X_test,y_test=splitData(data)
    N_estimators=[20,50,100,200,300,500]
    test_scores=[]
    for N in N_estimators:
        clf=RandomForestRegressor(n_estimators=N,max_features=0.2)
        test_score=np.sqrt(-cross_val_score(clf,X_train,y_train,cv=5,scoring='neg_mean_squared_error'))
        test_scores.append(np.mean(test_score))
    plt.plot(N_estimators,test_scores)
    plt.title('N_estimators vs cv error')
    plt.xlabel('N_estimators')
    plt.ylabel('neg_mean_squared_error')
    plt.show()

# 交叉验证XGBoost
def xgb_train(data,param='max_depth'):
    X_train,y_train,X_test,y_test=splitData(data)
    Max_depths=[i for i in range(1,10)]
    N_estimators=[20,50,100,150,200,300,500,800]
    test_scores=[]
    # 树数参数
    if param=='n_estimators':
        for N in N_estimators:
            clf=XGBRegressor(max_depth=2,learning_rate=0.1, n_estimators=N)
            test_score=np.sqrt(-cross_val_score(clf,X_train,y_train,cv=10,scoring='neg_mean_squared_error'))
            test_scores.append(np.mean(test_score))
        plt.plot(N_estimators,test_scores)
        plt.xlabel('n_estimators')
        plt.title('N_estimators vs cv error')
    # 最大深度参数
    elif param=='max_depth':
        for M in Max_depths:
            clf=XGBRegressor(max_depth=M,learning_rate=0.1, n_estimators=50)
            test_score=np.sqrt(-cross_val_score(clf,X_train,y_train,cv=10,scoring='neg_mean_squared_error'))
            test_scores.append(np.mean(test_score))
        plt.plot(Max_depths,test_scores)
        plt.xlabel('max_depth')
        plt.title('Max_depth vs cv error')
    plt.ylabel('neg_mean_squared_error')	
    plt.show()

# LightGBM
def lgb_train(data):
    X_train,y_train,X_test,y_test=splitData(data)
    Max_depths=[2, 3, 5, 6, 7, 9, 12, 15, 17, 25]
    num_leaves=[]
    N_estimators=[20,50,100,150,200,300,500,800]
    test_scores=[]
    # lgb.LGBMRegressor
    # estimator = lgb.LGBMRegressor(num_leaves=31)
    # param_grid = {
    #     # 'learning_rate': [0.01, 0.1, 1],
    #     'n_estimators': [20,50,100,150,200,300,500,800]
    # }
    # gbm = GridSearchCV(estimator, param_grid)
    # gbm.fit(X_train, y_train)
    # print('Best parameters found by grid search are:', gbm.best_params_)

    params = {'num_leaves': 38,
          'min_data_in_leaf': 50,
          'objective': 'regression',
          'max_depth': -1,
          'learning_rate': 0.1,
          "min_sum_hessian_in_leaf": 6,
          "boosting": "gbdt",
          "feature_fraction": 0.9,
          "bagging_freq": 1,
          "bagging_fraction": 0.7,
          "bagging_seed": 11,
          "lambda_l1": 0.1,
          "verbosity": -1,
          "nthread": 4,
          'metric': 'mae',
          "random_state": 2019,
          # 'device': 'gpu'
          }

    X_train,y_train,X_test,y_test=splitData(data)
    N_estimators=[10,20,50,100,200,300,500]
    test_scores=[]
    for N in N_estimators:
        clf=lgb.LGBMRegressor(n_estimators=N,num_leaves=31,objective='regression',
                            learning_rate=0.1)
        test_score=np.sqrt(-cross_val_score(clf,X_train,y_train,cv=10,scoring='neg_mean_squared_error'))
        test_scores.append(np.mean(test_score))
    plt.plot(N_estimators,test_scores)
    plt.title('N_estimators vs cv error')
    plt.xlabel('N_estimators')
    plt.ylabel('neg_mean_squared_error')
    plt.show()

# 包装法
def warrper(data):
    X_train,y_train,X_test,y_test=splitData(data)
    data=data.drop(['ID','ESTA_TRAN_PRICE_TOTAL'],axis='columns')
    Y=np.log1p(data.pop('ESTA_TRAN_PRICE_UNIT'))
    names=data.columns.values.tolist()
    X=data.values
    n_features=[i for i in range(1,61)]
    test_scores=[]
    for i in n_features:
        clf=Ridge(7.5)
        # clf=RandomForestRegressor(n_estimators=200,max_features=0.2)
        # clf=XGBRegressor(max_depth=2,learning_rate=0.1, n_estimators=50)
        rfe = RFE(clf, n_features_to_select=i,step=1)
        rfe.fit(X,Y)
        test_scores.append(rfe.score(X,Y))
    plt.plot(n_features,test_scores)
    # plt.title('N_estimators vs cv error')
    # plt.xlabel('n_features_to_select')
    # plt.ylabel('neg_mean_squared_error')
    plt.show()
    # print("Features sorted by their rank:")
    # print(sorted(zip(rfe.ranking_, names)))

if __name__ == '__main__':
    data=load_data('E:\研究生\不动产估值\cleaned.csv')
    # warrper(data)
    xgb_train(data,'n_estimators')
    # ridge_train(data)
    # rf_train(data)
    # lgb_train(data)
    