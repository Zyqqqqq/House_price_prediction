import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

def load_data(path):
    data=pd.read_csv(path,encoding='gb18030')
    # data.drop(['ESTA_TRAN_PRICE_TOTAL'],axis='columns',inplace=True)
    return data

# 划分训练集、测试集
def splitData(data,test_size=0.2):
    train_set, test_set = train_test_split(data, test_size=test_size, random_state=42)
    y_train=np.log1p(train_set.pop('ESTA_TRAN_PRICE_UNIT'))
    X_train=train_set.values
    y_test=np.log1p(test_set.pop('ESTA_TRAN_PRICE_UNIT'))
    X_test=test_set.values
    return X_train,y_train,X_test,y_test

# 岭回归测试
def ridge_test(data):
    X_train,y_train,X_test,y_test=splitData(data)
    ridge=Ridge(alpha=7.5)
    ridge.fit(X_train,y_train)
    ridge_predict=ridge.predict(X_test)
    y_ridge=np.expm1(ridge_predict)
    print(ridge.score(X_train,y_train))
    df=pd.DataFrame(data={'true':np.expm1(y_test),'predict':y_ridge})
    df.to_csv('res_ridge.csv',encoding='gb18030')

# 随机森林测试
def rf_test(data):
    X_train,y_train,X_test,y_test=splitData(data)
    rf=RandomForestRegressor(n_estimators=200,max_features=0.2)
    rf.fit(X_train,y_train)
    rf_predict=rf.predict(X_test)
    y_rf=np.expm1(rf_predict)
    test_score=rf.score(X_train,y_train)
    print(test_score)
    df=pd.DataFrame(data={'true_value':np.expm1(y_test),'predict':y_rf})
    # print(df)
    df.to_csv('res_rf.csv',encoding='gb18030')

# XGBoost测试
def xgb_test(data):
    X_train,y_train,X_test,y_test=splitData(data)
    xgb = XGBRegressor(max_depth=2,learning_rate=0.15, n_estimators=500)
    # xgb = XGBRegressor()
    xgb.fit(X_train,y_train)
    print(xgb.score(X_train,y_train))
    y_xgb=np.expm1(xgb.predict(X_test))
    df=pd.DataFrame(data={'true_value':np.expm1(y_test),'predict':y_xgb})
    df.to_csv('res_xgb.csv',encoding='gb18030')

    # print((np.expm1(y_test)/y_xgb).mean())

if __name__ == '__main__':
    data=load_data('E:\研究生\不动产估值\cleaned.csv')
    # xgb_test(data)
    # rf_test(data)
    ridge_test(data)