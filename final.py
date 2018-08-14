# -*- coding: utf-8 -*-
"""
Created on Wed Jul 25 12:29:56 2018

@author: asus
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Jul 24 09:52:21 2018

@author: asus
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import average_precision_score 
from sklearn.grid_search import GridSearchCV

def time_to_seconds(t):
    t0 = pd.to_datetime('2017-02-01 00:00:00')
    delt = t-t0
    days = delt.days
    sec = delt.seconds
    return (days*86400+sec)

path_test = 'test/ord_testB.csv'
path_ord_train = './train/ord_train.csv'
path_ord_chaifen = './train/ord_chaifen.csv'
path_ord_zqroomstatus = './train/ord_zqroomstatus.csv'
path_ord_bkroomstatus = './train/ord_bkroomstatus.csv'
path_mroominfo = './train/mroominfo.csv'
path_mhotelinfo = './train/mhotelinfo.csv'
path_hotelinfo = './train/hotelinfo.csv'


print("loading data")
ord_train = pd.read_csv(path_ord_train)
ord_test = pd.read_csv(path_test,encoding ='gb2312')
ord_chaifen = pd.read_csv(path_ord_chaifen)
ord_zqroomstatus = pd.read_csv(path_ord_zqroomstatus)
ord_bkroomstatus = pd.read_csv(path_ord_bkroomstatus)
mroominfo = pd.read_csv(path_mroominfo)
mhotelinfo = pd.read_csv(path_mhotelinfo)
hotelinfo = pd.read_csv(path_hotelinfo)

print("coping with training data")

print("adding features")
tmp1 = pd.DataFrame(ord_train.masterbasicroomid)
tmp2 = pd.DataFrame(mroominfo[['masterbasicroomid','totalrooms']])
m = pd.merge(tmp1,tmp2,'left')
m.rename(columns = {'totalrooms':'mtotal_room'},inplace = True)
ord_train = pd.concat([ord_train,m.mtotal_room],1,'inner')

tmp1 = pd.DataFrame(ord_train.masterhotelid)
tmp2 = pd.DataFrame(mhotelinfo[['masterhotelid','star']])
m = pd.merge(tmp1,tmp2,'left')
m.rename(columns = {'star':'mhotel_star'},inplace = True)
ord_train = pd.concat([ord_train,m.mhotel_star],1,'inner')

tmp1 = pd.DataFrame(ord_train.hotel)
tmp2 = pd.DataFrame(hotelinfo[['hotel','totalrooms']])
m = pd.merge(tmp1,tmp2,'left')
m.rename(columns = {'totalrooms':'total_room'},inplace = True)
ord_train = pd.concat([ord_train,m.total_room],1,'inner')

print("coping with nan")
del ord_train['confirmdate']
del ord_train['zone']
del ord_train['etd']
del ord_train['masterbasicroomid']
del ord_train['masterhotelid']
del ord_train['hotel']

col = ord_train.columns
for name in col:
    if ord_train[name].isnull().any():
        ord_train[name] = ord_train[name].fillna(-1)
        ind = ord_train[(ord_train[name] == -1)].index.tolist()
        ord_train = ord_train.drop(ind)
        
print("trainsforming datetime.....")
ord_train.orderdate = pd.to_datetime(ord_train.orderdate,infer_datetime_format = True)
ord_train.orderdate = np.array(ord_train.orderdate.apply(time_to_seconds,1)).reshape(-1,1)
ord_train.arrival = pd.to_datetime(ord_train.arrival,infer_datetime_format = True)
ord_train.arrival = np.array(ord_train.arrival.apply(time_to_seconds,1)).reshape(-1,1)

print("labeling.....")
for name in ['isholdroom','hotelbelongto','isebookinghtl','supplierchannel']:
    ord_train[name] = LabelEncoder().fit(ord_train[name]).transform(ord_train[name])
    

print("constructing training dataset.....")    
feature_name = ['orderdate','city','room','isholdroom','arrival','ordadvanceday',
                'supplierid','isvendor','hotelbelongto','isebookinghtl',
            'hotelstar','supplierchannel','mtotal_room','mhotel_star','total_room']

features = []
for name in feature_name:
    tmp = np.array(ord_train[name]).reshape(-1,1)
    features.append(tmp)

features = np.concatenate(features,1)
label = ord_train['noroom']

(training_inputs,testing_inputs,training_classes,testing_classes) = train_test_split(features, label, train_size=0.75, random_state=1)
decision_tree = DecisionTreeRegressor(criterion = "friedman_mse",min_samples_split = 20,min_samples_leaf = 20)
parameter_grid = {'min_samples_split': [50],
                  'min_samples_leaf': [50]}

grid_search = GridSearchCV(decision_tree,
                           param_grid=parameter_grid,
                           scoring='average_precision',
                           cv=5)

grid_search.fit(features,label)
best_params = grid_search.best_params_
print('Best score: {}'.format(grid_search.best_score_))
print('Best parameters: {}'.format(grid_search.best_params_))
decision_tree = DecisionTreeRegressor(criterion = "friedman_mse",
                                      min_samples_split = best_params['min_samples_split'],
                                      min_samples_leaf = best_params['min_samples_leaf'])
decision_tree.fit(features,label)

print("coping with testing data")

print("adding features")
tmp1 = pd.DataFrame(ord_test.masterbasicroomid)
tmp2 = pd.DataFrame(mroominfo[['masterbasicroomid','totalrooms']])
m = pd.merge(tmp1,tmp2,'left')
m.rename(columns = {'totalrooms':'mtotal_room'},inplace = True)
ord_test = pd.concat([ord_test,m.mtotal_room],1,'inner')

tmp1 = pd.DataFrame(ord_test.masterhotelid)
tmp2 = pd.DataFrame(mhotelinfo[['masterhotelid','star']])
m = pd.merge(tmp1,tmp2,'left')
m.rename(columns = {'star':'mhotel_star'},inplace = True)
ord_test = pd.concat([ord_test,m.mhotel_star],1,'inner')

tmp1 = pd.DataFrame(ord_test.hotel)
tmp2 = pd.DataFrame(hotelinfo[['hotel','totalrooms']])
m = pd.merge(tmp1,tmp2,'left')
m.rename(columns = {'totalrooms':'total_room'},inplace = True)
ord_test = pd.concat([ord_test,m.total_room],1,'inner')

print("coping with nan")
del ord_test['zone']
del ord_test['etd']
del ord_test['masterbasicroomid']
del ord_test['masterhotelid']
del ord_test['hotel']

col = ord_test.columns
for name in col:
    if ord_test[name].isnull().any():
        mean = ord_test[name].mean()
        ord_test[name] = ord_test[name].fillna(mean)
        
print("trainsforming datetime.....")
ord_test.orderdate = pd.to_datetime(ord_test.orderdate,infer_datetime_format = True)
ord_test.orderdate = np.array(ord_test.orderdate.apply(time_to_seconds,1)).reshape(-1,1)
ord_test.arrival = pd.to_datetime(ord_test.arrival,infer_datetime_format = True)
ord_test.arrival = np.array(ord_test.arrival.apply(time_to_seconds,1)).reshape(-1,1)

print("labeling.....")
for name in ['isholdroom','hotelbelongto','isebookinghtl','supplierchannel']:
    ord_test[name] = LabelEncoder().fit(ord_test[name]).transform(ord_test[name])

test_features = []
for name in feature_name:
    tmp = np.array(ord_test[name]).reshape(-1,1)
    test_features.append(tmp)
    
test_features = np.concatenate(test_features,1)

test_pred = decision_tree.predict(test_features)

ord_test = pd.read_csv(path_test,encoding ='gb2312')
outputs = pd.DataFrame({'orderid':ord_test['orderid'],
                                           'room':ord_test['room'],
                                           'arrival':ord_test['arrival'],
                                           'noroom':test_pred})
    

outputs.to_csv('testB.csv',index = False)
