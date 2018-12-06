#Build basic structure of network
import data_process
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

# This is the R_squared function that used for loss evaluation
def R_squared(y_true, y_pred):
    u = K.sum(K.square(y_true - y_pred))
    v = K.sum(K.square(y_true - K.mean(y_true)))
    return (1 - u/(v + K.epsilon()))

def make_combo(option1=['tanh','relu','linear'],
                option2=['0.1','0.01','0.001']):
    '''
    This functions generate list for all combinations of different hyperparameter
    It takes list of options of every hyperparameter as parameter
    It returns the list of combinations
    '''
    parameter_combo = []
    for i in option1:
        for j in option2:
            combo = []
            combo.append(i)
            combo.append(j)
            parameter_combo.append(combo)

    return parameter_combo

def make_pairwise_list(max_depth=2, options=['tanh', 'softmax', 'relu']):
    '''
    This functions makes list for all combinations of hyperparameters used for every layer
    It takes the number of layers and list of all combinations of different hyperparameter as parameters
    It returns the list of hyperparamter used for every iteration
    '''
    combinations = []
    for i in range(len(options)**max_depth):
        state = []
        for depth in range(max_depth):
            if depth == 0:
                state.append(options[i // len(options)**(max_depth-1)])
            elif depth == max_depth - 1:
                state.append(options[i % len(options)])
            else:
                state.append(options[i // len(options)**(depth) % len(options)])
        combinations.append(state)
    return combinations

def model_initial_search(data,test_fraction,random_state,layers,cumulative_time,params):
    filename = data["filename"]
    X_var = data["X_var"]
    Y_var = data["Y_var"]
    X,Y = read_file(filename,X_var,Y_var)
    input_dim = len(X_var)
    output_dim = len(Y_var)
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
                collection_folder = './collection%d' % (layers)
                if not os.path.exists(output_folder):
                    os.makedirs(output_folder)
                filepath=output_folder+"/weights-{epoch:02d}-{val_R_squared:.2f}.hdf5"
                checkpoint = keras.callbacks.ModelCheckpoint(filepath, monitor='val_R_squared', verbose=1, save_best_only=False, save_weights_only=True, mode='auto', period=10)
                callbacks_list = [earlystop,checkpoint]
                start = time.time()
                history = model.fit(X_train,Y_train,epochs=300, batch_size=10,callbacks=callbacks_list,validation_split=0.2,verbose=0)
                end = time.time()
                cumulative_time += (end-start)
                scores = model.evaluate(X_test,Y_test,verbose=0)
                iteration_n += 1
                if not os.path.exists("Results%d.txt"%(layers)):
                    f = open("Results%d.txt"%(layers),"w+")
                else:
                    f = open("Results%d.txt"%(layers),"a+")
                f.write("For this combination %s, R is %0.2f\r\n" %(parameter_list,scores[1]))
                if scores[1]>best_R:
                    best_param = parameter_list
                    best_R = scores[1]
                else:
                    pass
                f.write("The best_R for now is %0.4f and the combination is %s in %0.2f seconds" % (best_R,parameter_list,cumulative_time))
                x ={"layer_number":layers,"starting_n":iteration_n-1,"best_R":best_R,"best_param":best_param,"cumulative_time":cumulative_time}
                print(x)
                check_write(x,'latest.json')
        print("")
    f.close()
    print(best_param)
    print(best_R)
    print('model took %0.2f seconds to train'%(cumulative_time))
    return best_param,best_R

def model_continue_search(data,test_fraction,random_state,cumulative_time,params):
    filename = data["filename"]
    X_var = data["X_var"]
    Y_var = data["Y_var"]
    X,Y = read_file(filename,X_var,Y_var)
    input_dim = len(X_var)
    output_dim = len(Y_var)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_fraction, random_state=random_state)
    activation_functions = params["activation_functions"]
    units = params["units"]
    y1 = check_read("latest.json")
    starting_n = y1["starting_n"]
    print(y1)
    best_R = y1["best_R"]
    best_param = y1["best_param"]
    cumulative_time = y1["cumulative_time"]
    layers = y1["layer_number"]
    epoch_num = get_epoch_num(layers)
    iterations = (len(units)*len(activation_functions))**(layers+1)*len(activation_functions)
    inner_iterations = (len(units)*len(activation_functions))**layers
    options= make_combo(option1=activation_functions,option2=units)
    af_combs = make_pairwise_list(max_depth=layers, options=options)
    print(f'{layers}\t{options}\t{iterations} iterations required')
    iteration_n = 1
    run_once = 0
    for inner_iteration in range(inner_iterations):
        for option_in in options:
            inner_list=[]
            for k in range(layers):
                inner_list.append(af_combs[inner_iteration][k])
            for activation_out in activation_functions:
                if run_once == 1:
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
                    callbacks_list = model_creation_run_two(earlystop,iteration_n,layers)
                    start = time.time()
                    history = model.fit(X_train,Y_train,epochs=300, batch_size=10,callbacks=callbacks_list,validation_split=0.2,verbose=0)
                    end = time.time()
                    cumulative_time += (end-start)
                    best_param,best_R = model_creation_run_three(model,X_test,Y_test,iteration_n,parameter_list,layers,best_R,best_param)
                    iteration_n += 1
                    x ={"layer_number":layers,"starting_n":iteration_n-1,"best_R":best_R,"best_param":best_param,"cumulative_time":cumulative_time}
                    print(x)
                    check_write(x,'latest.json')
                    print("")
                else:
                    if not (iteration_n > starting_n):
                        iteration_n += 1
                        pass
                    else:
