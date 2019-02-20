#Build basic structure of network
import glob
import os
import time
import json
import data_process as dp
import param_record as pr
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
                if not os.path.exists("./Results/results%d.txt"%(layers)):
                    f = open("./Results/results%d.txt"%(layers),"w+")
                else:
                    f = open("results%d.txt"%(layers),"a+")
                f.write("For this combination %s, R is %0.2f\r\n" %(parameter_list,scores[1]))
                if scores[1]>best_R:
                    best_param = parameter_list
                    best_R = scores[1]
                else:
                    pass
                f.write("The best_R for now is %0.4f and combination is %s "% (best_R,best_param))
                iteration_n += 1
                x ={"layer_number":layers,"starting_n":iteration_n-1,"best_R":best_R,"best_param":best_param,"cumulative_time":cumulative_time}
                print(x)
                pr.check_write(x,'latest.json')
                print("")
        print("")
    f.close()
    print(best_param)
    print(best_R)
    print('Training process has been finished')
    print('model took %0.2f seconds to train'%(cumulative_time))
    return best_param,best_R

def model_continue_search(data,test_fraction,random_state,cumulative_time,params):
    X,Y,input_dim,output_dim = dp.data_info(data)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_fraction, random_state=random_state)
    activation_functions = params["activation_functions"]
    units = params["units"]
    activation_functions = params["activation_functions"]
    units = params["units"]
    y1 =pr.check_read("latest.json")
    starting_n = y1["starting_n"]
    print(y1)
    best_R = y1["best_R"]
    best_param = y1["best_param"]
    cumulative_time = y1["cumulative_time"]
    layers = y1["layer_number"]
    epoch_num = pr.get_epoch_num(layers)
    iterations = (len(units)*len(activation_functions))**(layers+1)*len(activation_functions)-starting_n
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
                    if not os.path.exists("results%d.txt"%(layers)):
                        f = open("results%d.txt"%(layers),"w+")
                    else:
                        f = open("results%d.txt"%(layers),"a+")
                    f.write("For this combination %s, R is %0.2f\r\n" %(parameter_list,scores[1]))
                    if scores[1]>best_R:
                        best_param = parameter_list
                        best_R = scores[1]
                    else:
                        pass
                    f.write("The best_R for now is %0.4f and combination is %s "% (best_R,best_param))
                    iteration_n += 1
                    x ={"layer_number":layers,"starting_n":iteration_n-1,"best_R":best_R,"best_param":best_param,"cumulative_time":cumulative_time}
                    print(x)
                    pr.check_write(x,'latest.json')
                    print("")
                else:
                    if not (iteration_n > starting_n):
                        iteration_n += 1
                        pass
                    else:
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
                        output_folder = './Results/collection%d/intermediate_output%d' % (layers,iteration_n)
                        file_ini = output_folder+'/weights-'+str(epoch_num)+'*'
                        filename = glob.glob(file_ini)
                        print(filename)
                        if os.path.isfile(filename[0]):
                            model.load_weights(filename[0])
                        else:
                            print("%s does not exists" % (filename[0]))
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
                        history = model.fit(X_train,Y_train,epochs=300,batch_size=10,callbacks=callbacks_list,validation_split=0.2,initial_epoch=epoch_num+1,verbose=0)
                        end = time.time()
                        cumulative_time += (end-start)
                        print('it already took %0.2f seconds' % (cumulative_time))
                        scores = model.evaluate(X_test,Y_test,verbose=0)
                        if not os.path.exists("./Results/results%d.txt"%(layers)):
                            f = open("./Results/results%d.txt"%(layers),"w+")
                        else:
                            f = open("./Results/results%d.txt"%(layers),"a+")
                        f.write("For this combination %s, R is %0.2f\r\n" %(parameter_list,scores[1]))
                        if scores[1]>best_R:
                            best_param = parameter_list
                            best_R = scores[1]
                        else:
                            pass
                        f.write("The best_R for now is %0.4f and combination is %s "% (best_R,best_param))
                        run_once = 1
                        iteration_n += 1
                        x ={"layer_number":layers,"starting_n":iteration_n-1,"best_R":best_R,"best_param":best_param,"cumulative_time":cumulative_time}
                        print(x)
                        pr.check_write(x,'latest.json')
                        print("")

    print(best_param)
    print(best_R)
    print('Training process has been finished')
    print('model took %0.2f seconds to train'%(cumulative_time))
    return best_param,best_R

def layer_search(data,test_fraction,random_state,cumulative_time,params):
    hidden_layers = params["hidden_layers"]
    best_list = []
    best_R_list = []
    entire_iterations = 0
    activation_functions = params["activation_functions"]
    units = params ["units"]
    for i in hidden_layers:
        iterations = (len(units)*len(activation_functions))**(i+1)*len(activation_functions)
        entire_iterations += iterations
    print(f"Complete search takes {entire_iterations} iterations to finish")
    for layers in hidden_layers:
        best_param,best_R = model_initial_search(data,test_fraction,random_state,layers,cumulative_time,params)
        print("best R for the combination %s with %d hidden layer is %0.4f" % (best_param,layers,best_R))
        best_list.append(best_param)
        best_R_list.append(best_R)
        update_list = {"best_list":best_list,"best_R_list":best_R_list}
        print(update_list)
        pr.check_write(update_list,"list.json")
    print(best_list,best_R_list)
    max_R = best_R_list[0]
    max_param = best_list[0]
    for i in range(len(best_R_list)-1):
        if best_R_list[i+1] > best_R_list[i]:
            max_R = best_R_list[i+1]
            max_param = best_list[i+1]
        else:
            pass
    return max_R,max_param

def continue_layer_search(data,test_fraction,random_state,cumulative_time,params):
    X,Y,input_dim,output_dim = dp.data_info(data)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_fraction, random_state=random_state)
    activation_functions = params["activation_functions"]
    units = params["units"]
    activation_functions = params["activation_functions"]
    units = params["units"]
    hidden_layers = params["hidden_layers"]
    y1 = pr.check_read("latest.json")
    layer_num = y1["layer_number"]
    epoch_num = pr.get_epoch_num(layer_num)
    if os.path.exists("./list.json"):
        y2 = pr.check_read("list.json")
        best_list = y2["best_list"]
        best_R_list = y2["best_R_list"]
    else:
        best_list = []
        best_R_list = []
    for layer in hidden_layers:
        if layer < layer_num:
            pass
        elif layer == layer_num:
            best_param,best_R=model_continue_search(data,test_fraction,random_state,cumulative_time,params)
            print("best R for the combination %s with %d hidden layer is %0.4f" % (best_param,layer,best_R))
            best_list.append(best_param)
            best_R_list.append(best_R)
        else:
            best_param,best_R = model_initial_search(data,test_fraction,random_state,layer,cumulative_time,params)
            print("best R for the combination %s with %d hidden layer is %0.4f" % (best_param,layer,best_R))
            best_list.append(best_param)
            best_R_list.append(best_R)
    print(best_list,best_R_list)
    max_R = best_R_list[0]
    max_param = best_list[0]
    for i in range(len(best_R_list)-1):
        if best_R_list[i+1] > best_R_list[i]:
            max_R = best_R_list[i+1]
            max_param = best_list[i+1]
        else:
            pass
    return max_R,max_param
                                                                                                                                                                                                 
