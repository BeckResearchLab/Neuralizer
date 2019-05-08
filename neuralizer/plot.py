# Draw plots for fitting results
import matplotlib.pyplot as plt
import data_process as dp
import keras
from keras import *
import keras.backend as K
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
import param_record as pr 

def R_squared(y_true, y_pred):
    u = K.sum(K.square(y_true - y_pred))
    v = K.sum(K.square(y_true - K.mean(y_true)))
    return (1 - u/(v + K.epsilon()))

def best_prediction(data,best_result):
	X,Y,input_dim,output_dim = dp.data_info(data)
	X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
	proj_name = data['proj_name']
	best_result = pr.check_read("%s_latest.json"%(proj_name))
	n = best_result['layer_number']
	best_param = best_result['best_param']
	print(best_param)
	model = keras.Sequential()
	model.add(keras.layers.Dense(best_param[0][1],input_dim = input_dim,activation=best_param[0][0]))
	for i in range(n-2):
		model.add(keras.layers.BatchNormalization(momentum=0.9))
		model.add(keras.layers.Dense(best_param[i+1][1],activation=best_param[i+1][0]))
	model.add(keras.layers.Dense(output_dim,activation=best_param[n-1][0]))
	model.compile(loss='mean_squared_error', optimizer='adam',metrics=[R_squared])
	earlystop = keras.callbacks.EarlyStopping(monitor='val_R_squared',min_delta=0.0001,patience=20,mode='auto')
	callbacks_list = [earlystop]
	model.fit(X_train,Y_train,epochs=300, batch_size=10,callbacks=callbacks_list,validation_split=0.2,verbose=0)
	Y_new = model.predict(X_test)

	return X_test,Y_test,Y_new