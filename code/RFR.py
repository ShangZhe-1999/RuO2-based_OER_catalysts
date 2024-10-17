# -*- coding: utf-8 -*-
"""
@Time ： 2023/10/25 15:38
@Auth ： Shang Zhe
@File ：RFR.py
@IDE ：PyCharm
"""
import os
import random

import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, mean_absolute_error,r2_score # RMSE
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
import time
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

path = "D:\\Desktop\\OER"

# RFR model begin
df = data_training = pd.read_csv(os.path.join(path,"infile"),sep=" ")
features = df.iloc[:,0:-1]
target = df.iloc[:, -1]
feat_labels = df.columns[0:-1]
#print(features)
#print(target)

seed = random.randint(0,1e6)

X_train, X_test, y_train, y_test = train_test_split(features,target,train_size= 0.9, random_state=seed)
RFR = RandomForestRegressor(n_estimators=200, max_depth=10)

kfold = KFold(n_splits=5, shuffle=False)
scores = cross_val_score(RFR, X_train, y_train,scoring='neg_mean_squared_error', cv=kfold)
mse_ads = [abs(s) for s in scores]
mse_tr = np.mean(mse_ads)
rmse_scores = [np.sqrt(abs(s)) for s in scores]

RFR.fit(X_train, y_train)

print("=================training======================")
y_train_pred = RFR.predict(X_train)
MAE = mean_absolute_error(y_train_pred, y_train)
MSE = mean_squared_error(y_train_pred, y_train)
RMSE = MSE ** 0.5
print ('MAE of training set {:.4f} eV'.format(MAE))
print ('RMSE of training set {:.4f} eV'.format(RMSE))
R_squr_train = r2_score(y_train,y_train_pred)
print ("R2 {:4f}".format(R_squr_train))
indices = y_train.index.values
#print(indices)
with open(os.path.join(path,"outfile1"),"w") as outfile:
    for i in range(len(indices)):
        print(y_train[indices[i]],y_train_pred[i],file=outfile)



print("=================test======================")
y_test_pred = RFR.predict(X_test)
MAE = mean_absolute_error(y_test_pred, y_test)
MSE = mean_squared_error(y_test_pred, y_test)
RMSE = MSE ** 0.5
print ('MAE of test set {:.4f} eV'.format(MAE))
print ('RMSE of test set {:.4f} eV'.format(RMSE))
R_squr_test = r2_score(y_test,y_test_pred)
print ("R2 {:4f}".format(R_squr_test))
indices = y_test.index.values
with open(os.path.join(path,"outfile2"),"w") as outfile:
    for i in range(len(indices)):
        print(y_test[indices[i]],y_test_pred[i],file=outfile)


importances = RFR.feature_importances_
indices = np.argsort(importances)[::-1]
for i in range(X_train.shape[1]):
    print("%2d) %-*s %f" % (i + 1, 30, feat_labels[indices[i]], importances[indices[i]]))