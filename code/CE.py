# -*- coding: utf-8 -*-
"""
@Time ： 2023/10/24 18:14
@Auth ： Shang Zhe
@File ：19.py
@IDE ：PyCharm
"""
from icet import ClusterSpace, StructureContainer, ClusterExpansion
from icet.core.local_orbit_list_generator import LocalOrbitListGenerator
from icet.core.structure import Structure
from icet.tools import enumerate_structures
from sklearn.linear_model import LassoCV, Lasso, Ridge, RidgeCV
from sklearn.model_selection import train_test_split, cross_val_predict, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error,r2_score
from sklearn.metrics import *
from ase.io import read
import os
import pandas as pd
import numpy as np
import random
import time

path = os.getcwd()
start_time = time.time()

def RuDataset():
    db = read(os.path.join(path,"structure.json"),":")
    dft_data = pd.read_csv(os.path.join(path,"mixingenergy.csv"))
    indices = dft_data.index.values
    DataFrame = pd.DataFrame(columns=['atoms','mixingenergy'])
    for idx,i in enumerate(indices):
        DataFrame.loc[idx] = [db[idx], dft_data.loc[i,'mixingenergy']]
    return DataFrame

df = RuDataset()

for i in df.index.values:
    positions = df.loc[i,"atoms"].positions
    sorted = np.lexsort((positions[:, 2], positions[:, 1], positions[:, 0]))
    df.at[i, 'atoms'] = df.loc[i, 'atoms'][sorted]

position = df.loc[1,"atoms"].positions

# Remove pure phases since there aren't many of them to learn from
df = df[df['mixingenergy']!=0.0]

# Find list of allowed elements for every site
chemsymbols = df.loc[1,'atoms'].get_chemical_symbols()
allowed_elements = ["Sc",'Ti',"V",'Cr','Mn','Fe',"Co",'Ni','Cu','Zn','Ga','Ge',
                    'Y','Zr','Nb','Mo','Tc','Ru','Rh','Pd','Ag','Cd','In','Sn','Sb',
                    'La','Hf','Ta','W','Re','Os','Ir','Pt','Au','Hg','Tl','Pb','Bi','Po']
substitutions = []
for chem in chemsymbols:
    if chem != "O":
        substitutions.append(allowed_elements)
    else:
        substitutions.append(["O"])

print("==========================Cluster Expansion=========================")
print("1. Preparing a cluster space")
cs= ClusterSpace(structure=df.loc[1,"atoms"], cutoffs=[8,6], chemical_symbols=substitutions,symprec=1e-04,position_tolerance=1e-04)
print("2. Compiling a structure container")
sc = StructureContainer(cluster_space=cs)
for i in df.index.values:
    sc.add_structure(structure=df.loc[i,'atoms'],properties={'mixingenergy': df.loc[i,'mixingenergy']})


# RIDGECV model
seed = random.randint(1,1E06)
print("RIDGE_CV model")
# k-fold
model_RidgeCV1 = RidgeCV(fit_intercept=True,alphas=(0.01,0.1,1.0,10,100),cv=10)
y = sc.get_fit_data(key='mixingenergy')[1]
x = sc.get_fit_data(key='mixingenergy')[0]
model_RidgeCV1.fit(x,y)
crossmodel = Ridge(alpha=model_RidgeCV1.alpha_,fit_intercept=model_RidgeCV1.fit_intercept,max_iter=10000)
y_pred = cross_val_predict(crossmodel, x, y, cv=10)
score = cross_val_score(crossmodel,x, y, cv=10)
MAE = mean_absolute_error(y_pred, y)
MSE = mean_squared_error(y_pred, y)
RMSE = MSE ** 0.5
R_squr_test = r2_score(y,y_pred)
print ('RidgeCV k-fold validation MAE: {:.4f} eV/atom'.format(MAE/144))
print ('RidgeCV k-fold validation RMSE: {:.4f} eV/atom'.format(RMSE/144))
print ("RidgeCV k-fold validation R^2: {:4f}".format(R_squr_test))

# regular training
y = sc.get_fit_data(key='mixingenergy')[1]
x = sc.get_fit_data(key='mixingenergy')[0]
X_train, X_test, y_train, y_test = train_test_split(x,y,train_size= 0.9, random_state=seed)
model_RidgeCV2 = RidgeCV(fit_intercept=True,alphas=(0.01,0.1,1.0,10,100),cv=10)
model_RidgeCV2.fit(x,y)
y_train_pred = model_RidgeCV2.predict(X_train)
y_test_pred = model_RidgeCV2.predict(X_test)
MAE = mean_absolute_error(y_test_pred, y_test)
MSE = mean_squared_error(y_test_pred, y_test)
RMSE = MSE ** 0.5
R_squr_test = r2_score(y_test,y_test_pred)
print ('RIDGE test MAE: {:.4f} eV/atom'.format(MAE/144))
print ('RIDGE test RMSE: {:.4f} eV/atom'.format(RMSE/144))
print ("RIDGE test R^2: {:.4f}".format(R_squr_test))
with open(os.path.join(path,"predtrain"),"w") as outfile1, open(
    os.path.join(path,"predtest"),"w") as outfile2:
    for i in range(len(y_train)):
        print(y_train[i],y_train_pred[i],file=outfile1)
    for i in range(len(y_test)):
        print(y_test[i],y_test_pred[i],file=outfile2)

print("4. Finalizing the cluster expansion")
intercept_mixing = model_RidgeCV2.intercept_
print(intercept_mixing)
ce = ClusterExpansion(cluster_space=cs, parameters=model_RidgeCV2.coef_)
ce.write("ex.model")

print("--- %s seconds ---" % (time.time() - start_time))

