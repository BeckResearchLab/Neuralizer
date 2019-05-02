# Draw plots for fitting results
import model_simplified_search as mss
import matplotlib.pyplot as plt
import data_process as dp
import keras
from keras import *
import keras.backend as K
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split

def R_squared(y_true, y_pred):
    u = K.sum(K.square(y_true - y_pred))
    v = K.sum(K.square(y_true - K.mean(y_true)))
    return (1 - u/(v + K.epsilon()))

def best_prediction(data):
	X,Y,input_dim,output_dim = dp.data_info(data)
	X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
	model = keras.Sequential()
	model.add(keras.layers.Dense(10,input_dim = input_dim,activation="linear"))
	model.add(keras.layers.BatchNormalization(momentum=0.9))
	model.add(keras.layers.Dense(18,activation="tanh"))
	model.add(keras.layers.BatchNormalization(momentum=0.9))
	model.add(keras.layers.Dense(18,activation="tanh"))
	model.add(keras.layers.Dense(output_dim,activation='tanh'))
	model.compile(loss='mean_squared_error', optimizer='adam',metrics=[R_squared])
	earlystop = keras.callbacks.EarlyStopping(monitor='val_R_squared',min_delta=0.0001,patience=20,mode='auto')
	callbacks_list = [earlystop]
	model.fit(X_train,Y_train,epochs=300, batch_size=10,callbacks=callbacks_list,validation_split=0.2,verbose=0)
	Y_new = model.predict(X_test)

	return X_test,Y_test,Y_new