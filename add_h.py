import numpy as np
import pandas as pd
import requests
import time

def get_info(df,i,k,value,key,radius=1000):
    base = 'https://restapi.amap.com/v3/place/around?parameters'
    par = {'location': df.loc[i,'LOCATION'], 'key': key,'types':value,'radius':radius}
    response = requests.get(base, par)
    answer = response.json()
    # print(answer)
    df.loc[i,k+'({d}m)'.format(d=radius)]=answer['count']

# 住宅的影响因素
inf_house={'DOWNTOWN':{'code':'190500','distance':2000},'RESTAURANT':{'code':'050000','distance':1000},
    # 商场 
      'SHOPS':{'code':'060100','distance':1000},'CONVENIENCE_STORE':{'code':'060200','distance':1000},
      'SUPERMARKET':{'code':'060400','distance':1000},'COMPREHENSICE_MARKET':{'code':'060700','distance':1000},
      'COMMERCIAL_STREET':{'code':'061000','distance':1000},'FRANCHISE_STORE':{'code':'061200','distance':1000},
    # 公园广场
      'PARKS':{'code':'110101','distance':1500},'SQUARE':{'code':'110105','distance':1000},
    # 学校
      'KINDERGARTEN':{'code':'141204','distance':1000},'ELEMENTARY_S':{'code':'141203','distance':1000},
      'MIDDLE_S':{'code':'141202','distance':1500},'COLLEGE':{'code':'141201','distance':2000},
      'TRAINING':{'code':'141400','distance':1000},
    # 银行
      'BANKS':{'code':'160100','distance':1000},'ATM':{'code':'160300','distance':1000},
    # 医院  
      'AAA_HOSPITAL':{'code':'090101','distance':1000},'HEALTH_CENTER':{'code':'090102','distance':1000},
      'SPECIAL_HOSPITAL':{'code':'090200','distance':1000},'CLINIC':{'code':'090300','distance':1000},
      'EMERGENCY_CENTER':{'code':'090400','distance':1000},'PHARMACY':{'code':'090601','distance':1000},
      'VET':{'code':'090700','distance':1000},
    #  交通
      'SUBWAY_COUNT':{'code':'150500','distance':[1000,2000,3000,5000]},
      'BUS_COUNT':{'code':'150700','distance':[250,500,750,1000,2000]}}

# 别墅的影响因素
inf_villa={    
    # 银行
      'BANKS':{'code':'160100','distance':2000},'ATM':{'code':'160300','distance':2000},
    # 医院  
      'AAA_HOSPITAL':{'code':'090101','distance':5000},'HEALTH_CENTER':{'code':'090102','distance':5000},
      'SPECIAL_HOSPITAL':{'code':'090200','distance':5000},'CLINIC':{'code':'090300','distance':5000},
      'EMERGENCY_CENTER':{'code':'090400','distance':5000},'PHARMACY':{'code':'090601','distance':5000},
      'VET':{'code':'090700','distance':5000},
    # 商场
      'SHOPS':{'code':'060100','distance':3000},'CONVENIENCE_STORE':{'code':'060200','distance':1000},
      'COMPREHENSICE_MARKET':{'code':'060700','distance':5000},'SUPERMARKET':{'code':'060400','distance':5000},
    #  交通
      'SUBWAY_COUNT':{'code':'150500','distance':[1000,2000,5000]},'HIGHWAY':{'code':'190305','distance':5000},
      'BUS_COUNT':{'code':'150700','distance':[250,500,750,1000,2000,5000]},'URBAN_EXPRESSWAY':{'code':'190309','distance':5000},
    # 公园广场
      'PARKS':{'code':'110101','distance':5000},'SCENERY':{'code':'110200','distance':5000},
      'GOLF':{'code':'080200','distance':5000}
      }

# 商业影响因素
inf_business={

}
# 工业影响因素
inf_industry={

}
# 办公影响因素
inf_office={

}
# 其他影响因素

# 房屋影响因素
dict_house={'house':inf_house,'villa':inf_villa,'business':inf_business,'industry':inf_industry,
            'office':inf_office}

# 房屋数据扩充
def houseAdd(df,key):
    for i in range(2500,3000):
        if i%50==0:
            print(i)
        # print(i)
        # if df.loc[i,'ESTA_PROP']=='住宅':
        for k,value in dict_house['house'].items():
          # 多距离
          if isinstance(value['distance'],list): 
            for distance in value['distance']:
              get_info(df,i,k,value['code'],key,distance)
          else:
              get_info(df,i,k,value['code'],key,value['distance'])
        # elif df.loc[i,'ESTA_PROP']=='别墅':
        #   for k,value in dict_house['villa'].items():
        #     # 多距离
        #     if isinstance(value['distance'],list): 
        #         for distance in value['distance']:
        #             get_info(df,i,k,value['code'],key,distance)
        #     else:
        #         get_info(df,i,k,value['code'],key,value['distance'])

def houseTest(loc,key):
    base = 'https://restapi.amap.com/v3/place/around?parameters '
    par = {'location': loc, 'key': key,'types':'050000','radius':1000}
    response = requests.get(base, par)
    answer = response.json()
    print(answer['count'])
    print(len(answer['pois']))
    for res in answer['pois']:
        print(res['name'])

if __name__ == '__main__':
    df1=pd.read_csv('E:\研究生\不动产估值\sifa\PmHouse11.csv', encoding = 'gb18030',low_memory=False)
    # df1=df1.drop(['LXR','LXDH'],axis='columns')
    houseAdd(df1,'e36ead85996afabc3a72ca2cd0e54885')
    df1.to_csv('PmHouse12.csv', encoding = 'gb18030',index=False)
    print(df1.head())
    # houseTest('40.224396,116.23801','688023611cb918854fd6880ffcee10b7')

# 688023611cb918854fd6880ffcee10b7
# d78b6082ad0bfa146328415d32150e63  2
# e36ead85996afabc3a72ca2cd0e54885  uncle
# 69795538a97b6ce4dd13a868ea42ffc8  ww