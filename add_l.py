import numpy as np
import pandas as pd
import requests
import time
import timeit

def get_info(df,i,k,value,key):
    base = 'https://restapi.amap.com/v3/place/around?parameters '
    par = {'location': df.loc[i,'LOCATION'], 'key': key,'types':value,'radius':5000}
    response = requests.get(base, par)
    answer = response.json()
    df.loc[i,k]=answer['count']

# 工业用地
inf_industry={'AIRPORT':'150104','TRAIN':'150210','PORT':'150304','BORDER_CROSSING':'151000','FERRY':'151200',
              'FACTORY':'170300','FARM':'170400'}
# 住宅/综合/其他用地
inf_houseLand={'SUBWAY_COUNT':'150500','RESTAURANT':'050000','SHOPS':'060000|080600',
      'HOSPITALS':'090000','SCHOOLS':'141200','BUS_COUNT':'150700','BANKS':'160100'}
# 商业用地
inf_business={'RESTAURANT':'050000','SHOPS':'060000','SPORTS':'080000','THEATER&CINEMA':'080600',
       'TOURIST_ATTRACTION':'110000','COMMERCIAL_HOUSE':'120201|120203'}
# 土地影响因素
dict_land={'industry':inf_industry,'house':inf_houseLand,'business':inf_business}

# 土地数据扩充
def landAdd(df,key):
    for i in range(41000,len(df)):
        if i%200==0:
            print(i)
        if type(df.loc[i,'LOCATION'])!=str and np.isnan(df.loc[i,'LOCATION']):
            continue
        if df.loc[i,'LAND_PROP']=='工业用地':
            for k,value in dict_land['industry'].items():
                get_info(df,i,k,value,key)
        elif (df.loc[i,'LAND_PROP']=='住宅用地')|(df.loc[i,'LAND_PROP']=='综合用地(含住宅)'):
            for k,value in dict_land['house'].items():
                get_info(df,i,k,value,key)
        elif df.loc[i,'LAND_PROP']=='商业/办公用地':
            for k,value in dict_land['business'].items():
                get_info(df,i,k,value,key)  
        else:
            for k,value in dict_land['house'].items():
                get_info(df,i,k,value,key)

if __name__ == '__main__':
    df=pd.read_csv('E:\研究生\不动产估值\landRes7.csv', encoding = 'gb18030')
    landAdd(df,'688023611cb918854fd6880ffcee10b7')
    df.to_csv('landRes1.csv', encoding = 'gb18030',index=False)

# 688023611cb918854fd6880ffcee10b7  1
# d78b6082ad0bfa146328415d32150e63  2
# e36ead85996afabc3a72ca2cd0e54885  uncle
# 69795538a97b6ce4dd13a868ea42ffc8  ww
# 6ec98939596f2c0d79062952a5590a02  backup