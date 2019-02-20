# This module helps choose best paramter combinations on samll portion of data
import model_create as md
import data_process as dp
import glob
import os
import time
import json
import keras
from keras import *
import keras.backend as K
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def pca_eval(data):
    '''
    This function is to evaluate the rugness of data, whcih would determine the smaple size for pretraining process
    It takes formatted data as parameter
    The explained variance ratio is printed
    '''
    X_var,Y_var,input_dim,output_dim = dp.data_info(data)
    x = StandardScaler().fit_transform(X_var)
    pca = PCA()
    pca.fit(x)
    print(pca.explained_variance_ratio_)  
    return pca.explained_variance_ratio_

def pretrain_search(data,portion,test_fraction,random_state,layers,params):
    X,Y,input_dim,output_dim = dp.data_info(data)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_fraction, random_state=random_state)
    activation_functions = params["activation_functions"]
    units = params["units"]
    iterations = (len(units)*len(activation_functions))**(layers+1)*len(activation_functions)
    inner_iterations = (len(units)*len(activation_functions))**layers
    options= make_combo(option1=activation_functions,option2=units)
    af_combs = make_pairwise_list(max_depth=layers, options=options)
    print(f'{layers}\t{options}\t{iterations} iterations required')
    best_R = 0.0
    best_param = []
    iteration_n = 1
    for n in range(layers):
        best_param.append(['none','none'])
    for inner_iteration in range(inner_iterations):
        for option_in in options:
            inner_list=[]
            for k in range(layers):
                inner_list.append(af_combs[inner_iteration][k])
            for activation_out in activation_functions:
                print(inner_list)
                print(f"running iteration {iteration_n}")
                parameter_list = []
                parameter_list.append(option_in)
                parameter_list.extend(inner_list)
                parameter_list.append(activation_out)
                print(f"create input layer with activation of {option_in[0]} and units of {option_in[1]}")
                model = keras.Sequential()
                model.add(keras.layers.Dense(option_in[1],input_dim = input_dim,activation=option_in[0]))
                for i in range(len(inner_list)):
                    print(f"create hidden layer {i+1} of activation {inner_list[i][0]} and units {inner_list[i][1]}")
                    model.add(keras.layers.BatchNormalization(momentum=0.9))
                    model.add(keras.layers.Dense(inner_list[i][1],activation = inner_list[i][0]))
                print(f"create output layer with activation of {activation_out} and units of {output_dim}")
                model.add(keras.layers.Dense(output_dim,activation=activation_out))
                model.compile(loss='mean_squared_error', optimizer='adam',metrics=[R_squared])
                earlystop = keras.callbacks.EarlyStopping(monitor='val_R_squared',min_delta=0.0001,patience=20,mode='auto')
                collection_folder = './Results/collection%d' % (layers)
                if not os.path.exists(collection_folder):
                    os.makedirs(collection_folder)
                output_folder = './Results/collection%d/intermediate_output%d'%(layers,iteration_n)
                if not os.path.exists(output_folder):
                    os.makedirs(output_folder)
                filepath=output_folder+"/weights-{epoch:02d}-{val_R_squared:.2f}.hdf5"
                checkpoint = keras.callbacks.ModelCheckpoint(filepath, monitor='val_R_squared', verbose=1, save_best_only=False, save_weights_only=True, mode='auto', period=10)
                callbacks_list = [earlystop,checkpoint]
                start = time.time()
                history = model.fit(X_train,Y_train,epochs=300, batch_size=10,callbacks=callbacks_list,validation_split=0.2,verbose=0)
                end = time.time()
                cumulative_time += (end-start)
                print('it already took %0.2f seconds' % (cumulative_time))
                scores = model.evaluate(X_test,Y_test,verbose=0)

