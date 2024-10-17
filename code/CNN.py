# -*- coding: utf-8 -*-
"""
@Time ： 2024/4/18 15:26
@Auth ： Shang Zhe
@File ：CNN_overp.py
@IDE ：PyCharm
"""
import os.path
import random

import numpy as np
import pickle
import time
import argparse
import sys

# keras/sklearn libraries
import keras
from keras.preprocessing import sequence
from keras.models import Sequential, Model, load_model
from keras.optimizers import Adam
from keras.layers import Dense, Dropout, Activation, Input, Reshape, BatchNormalization
from keras.layers import (
    Conv1D,
    GlobalAveragePooling1D,
    MaxPooling1D,
    GlobalAveragePooling1D,
    Reshape,
    AveragePooling1D,
    Flatten,
    Concatenate,
)
from keras import backend
from keras.callbacks import TensorBoard, LearningRateScheduler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error,r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler


parser = argparse.ArgumentParser(description="ML framework")

parser.add_argument("--multi_adsorbate",default=2,type=int,help="train for single adsorbate (0) or multiple (1) (default: 0)",)
parser.add_argument("--run_mode",default=1,type=int,help="run regular (0) or 5-fold CV (1) (default: 0)",)
parser.add_argument("--split_ratio", default=0.1, type=float, help="train/test ratio (default:0.1)")
parser.add_argument("--epochs", default=50, type=int, help="number of total epochs to run (default:60)")
parser.add_argument("--batch_size", default=20, type=int, help="batch size (default:6)")
parser.add_argument("--channels", default=8, type=int, help="number of channels (default: 9)")
args = parser.parse_args(sys.argv[1:])

def decay_schedule(epoch, lr):
    # Simple learning rate scheduler
    if epoch == 0:
        lr = 0.001
    elif epoch == 15:
        lr = 0.0005
    elif epoch == 35:
        lr = 0.0001
    elif epoch == 45:
        lr = 0.00005
    elif epoch == 55:
        lr = 0.00001
    elif epoch == 65:
        lr = 0.000005
    elif epoch == 70:
        lr = 0.000001
    return lr

def load_data():
    """
    load data containing:
    (1) dos of surface
    (2) adsorption energy(target)
    """
    with open(os.path.join(path, "OER", "adsorption_energy_binary", "adsorptionOH"), "rb") as infile1, open(
        os.path.join(path, "OER", "adsorption_energy_binary", "adsorptionO"), "rb") as infile2, open(
        os.path.join(path, "OER", "adsorption_energy_binary", "adsorptionOOH"), "rb") as infile3:
        adsorptionOH, adsorptionO, adsorptionOOH = pickle.load(infile1), pickle.load(infile2), pickle.load(infile3)

    """
    args.multi_adsorbate == 0 DOS of substrate
    args.multi_adsorbate == 1 DOS of substrate and adsored O in the *O intermediate
    args.multi_adsorbate == 2 DOS of adsored O in the *O intermediate
    """
    if args.multi_adsorbate == 0:
        with open(os.path.join(path,"OER","dos_binary", "dos"), "rb") as infile4:
            dos = pickle.load(infile4)
        dos = dos[:, :, 0:args.channels]
        dos = dos.astype(np.float32)

    elif args.multi_adsorbate == 1:
        with open(os.path.join(path,"OER","dos_binary", "odos"), "rb") as infile5:
            odos = pickle.load(infile5)
        dos = odos[:, :, 0:args.channels]
        dos = dos.astype(np.float32)

    elif args.multi_adsorbate == 2:
        with open(os.path.join(path,"OER","dos_binary", "odos"), "rb") as infile6:
            odos = pickle.load(infile6)
        dos = odos[:, :, 9:args.channels+9]
        dos = dos.astype(np.float32)

    print("number of *OH : {}".format(len(adsorptionOH)))
    print("number of *O  : {}".format(len(adsorptionO)))
    print("number of *OOH: {}".format(len(adsorptionOOH)))
    print("number of dos : {}".format(len(dos)))
    print("shape of dos  : {}".format(dos.shape))

    return adsorptionOH,adsorptionO,adsorptionOOH,dos

def dos_featurizer(channels):
    input_dos = Input(shape=(2000, channels))
    x1 = AveragePooling1D(pool_size=4, strides=2, padding="same")(input_dos)
    x2 = AveragePooling1D(pool_size=25, strides=2, padding="same")(input_dos)
    x3 = AveragePooling1D(pool_size=100, strides=2, padding="same")(input_dos)
    x = Concatenate(axis=-1)([x1,x2,x3])
    x = Conv1D(50, 20, activation="relu", padding="same", strides=2)(x)
    x = BatchNormalization()(x)
    x = Conv1D(70, 3, activation="relu", padding="same", strides=2)(x)
    x = AveragePooling1D(pool_size=3, strides=2, padding="same")(x)
    x = Conv1D(100, 3, activation="relu", padding="same", strides=2)(x)
    x = AveragePooling1D(pool_size=3, strides=2, padding="same")(x)
    x = Conv1D(120, 3, activation="relu", padding="same", strides=2)(x)
    x = AveragePooling1D(pool_size=3, strides=2, padding="same")(x)
    x = Conv1D(150, 3, activation="relu", padding="same", strides=1)(x)
    shared_model = Model(input_dos, x)
    return shared_model

def create_model(shared_conv, channels):
    input1 = Input(shape=(2000, channels))
    conv1 = shared_conv(input1)
    convmerge = Concatenate(axis=-1)([conv1])
    convmerge = Flatten()(convmerge)
    convmerge = Dropout(0.2)(convmerge)
    # fully connected layer
    convmerge = Dense(200, activation="linear")(convmerge)
    convmerge = Dense(240, activation="relu")(convmerge)
    convmerge = Dense(240, activation="relu")(convmerge)
    out = Dense(1, activation="linear")(convmerge)
    model = Model([input1], out)
    return model

# regular training
def run_training(args, dos,adsorption):
    seed = random.randint(0, 1e+06)
    x_train, x_test, y_train, y_test = train_test_split(
            dos, adsorption, test_size=args.split_ratio, random_state=seed)
    x_validation, y_validation = x_test, y_test

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train.reshape(-1, x_train.shape[2])).reshape(x_train.shape)
    x_validation = scaler.fit_transform(x_validation.reshape(-1, x_validation.shape[2])).reshape(x_validation.shape)
    x_test = scaler.transform(x_test.reshape(-1, x_test.shape[2])).reshape(x_test.shape)

    shared_conv = dos_featurizer(args.channels)
    lr_scheduler = LearningRateScheduler(decay_schedule, verbose=0)
    tensorboard = TensorBoard(log_dir="logs/{}".format(time.time()), histogram_freq=1)
    model = create_model(shared_conv, args.channels)
    model.compile(loss="logcosh", optimizer=Adam(0.001), metrics=["mean_absolute_error"])
    model.summary()
    model.fit(x_train, y_train,batch_size=args.batch_size, epochs=args.epochs,
                  validation_data=(x_test,y_test),callbacks=[tensorboard, lr_scheduler])

    y_train_pred = model.predict(x_train)
    y_train_pred = y_train_pred.reshape(len(y_train_pred))
    y_validation_pred = model.predict(x_validation)
    y_validation_pred = y_validation_pred.reshape(len(y_validation_pred))
    y_test_pred = model.predict(x_test)
    y_test_pred = y_test_pred.reshape(len(y_test_pred))

    print("train R^2: ", r2_score(y_train, y_train_pred))
    print("train MAE: ", mean_absolute_error(y_train, y_train_pred))
    print("train RMSE: ", mean_squared_error(y_train, y_train_pred) ** (0.5))
    print("validation R^2: ", r2_score(y_validation,y_validation_pred))
    print("validation MAE: ", mean_absolute_error(y_validation, y_validation_pred))
    print("validation RMSE: ", mean_squared_error(y_validation, y_validation_pred) ** (0.5))
    print("test R^2: ", r2_score(y_test, y_test_pred))
    print("test MAE: ", mean_absolute_error(y_test, y_test_pred))
    print("test RMSE: ", mean_squared_error(y_test, y_test_pred) ** (0.5))

# kfold
def run_kfold(args,dos,adsorption):
    seed = random.randint(0, 1e+06)
    kfold = KFold(n_splits=10, shuffle=True, random_state=seed)

    cvscores, R2 = [], []
    for train, test in kfold.split(dos, adsorption):
        scaler_CV = StandardScaler()
        dos[train] = scaler_CV.fit_transform(dos[train].reshape(-1, dos[train].shape[-1])).reshape(dos[train].shape)
        dos[test] = scaler_CV.transform(dos[test].reshape(-1, dos[test].shape[-1])).reshape(dos[test].shape)

        keras.backend.clear_session()
        shared_conv = dos_featurizer(args.channels)
        lr_scheduler = LearningRateScheduler(decay_schedule, verbose=0)

        model_CV = create_model(shared_conv, args.channels)
        model_CV.compile(loss="logcosh", optimizer=Adam(0.001), metrics=["mean_absolute_error"])
        model_CV.fit(dos[train], adsorption[train],batch_size=args.batch_size,epochs=args.epochs,verbose=0,callbacks=[lr_scheduler],)
        scores = model_CV.evaluate(dos[test],adsorption[test],verbose=0,)
        y_test_pred_CV_temp = model_CV.predict(dos[test])
        y_test_pred_CV_temp = y_test_pred_CV_temp.reshape(len(y_test_pred_CV_temp))
        R2.append(r2_score(adsorption[test], y_test_pred_CV_temp))
        print((model_CV.metrics_names[1], scores[1]), R2[-1])
        cvscores.append(scores[1])
    print((np.mean(cvscores), np.std(cvscores)))
    print("kfold R^2: ", r2_score(y_test_CV, y_test_pred_CV))
    print("kfold MAE: ", mean_absolute_error(y_test_CV, y_test_pred_CV))
    print("kfold RMSE: ", mean_squared_error(y_test_CV, y_test_pred_CV) ** (0.5))


def main():
    start_time = time.time()
    adsorptionOH, adsorptionO, adsorptionOOH,dos = load_data(args.multi_adsorbate)
    dos, target = dos, adsorptionOOH

    if args.run_mode == 0:
        run_training(args,dos,target)
    elif args.run_mode == 1:
        run_kfold(args,dos,target)

    print("--- %s seconds ---" % (time.time() - start_time))

if __name__ == "__main__":
    path = "D:\\Desktop"
    main()
