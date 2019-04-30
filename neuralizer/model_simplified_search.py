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

def model_search(data,test_fraction,random_state,params,cumulative_time = 0.0,restart=False):
    """
        docstring goes here, Rainie!
    """
    
    X,Y,input_dim,output_dim = dp.data_info(data)
    proj_name = data['proj_name']
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_fraction, random_state=random_state)
    activation_functions = params["activation_functions"]
    units = params["units"]
    hidden_layers = params["hidden_layers"]

    
    if os.path.exists("./%s_list.json"%(proj_name)):
        y2 = pr.check_read("%s_list.json")%(proj_name)
        best_list = y2["best_list"]
        best_R_list = y2["best_R_list"]
    else:
        best_list = []
        best_R_list = []
    
    total_iteration = 0
    for i in hidden_layers:
        total_iteration += (len(units)*len(activation_functions))**(i+1)*len(activation_functions)

    for layers in hidden_layers:
        iteration_n = 1
        iteration_l = 0
        options= make_combo(option1=activation_functions,option2=units)
        af_combs = make_pairwise_list(max_depth=layers, options=options)
        if restart:
            y1 = pr.check_read("%s_latest.json"%(proj_name))
            starting_n = y1["starting_n"]
            print(y1)
            best_R = y1["best_R"]
            best_param = y1["best_param"]
            cumulative_time = y1["cumulative_time"]
            layer_num = y1["layer_number"]
            epoch_num = pr.get_epoch_num(layer_num) - 10
            run_once = 0
        else:
            best_param = []
            for n in range(layers):
                best_param.append(['none','none'])
 
            best_R = 0.0
            layer_num = layers
            run_once = 1
            starting_n = 1

        
        if layers < layer_num:
            iteration_l += (len(units)*len(activation_functions))**(layers)*len(activation_functions)
            total_iteration -= iteration_l
            pass 
        else:
            inner_iterations = (len(units)*len(activation_functions))**layers
            for inner_iteration in range(inner_iterations):
                for option_in in options:
                    inner_list=[]
                    for k in range(layers):
                        inner_list.append(af_combs[inner_iteration][k])
                    for activation_out in activation_functions:
                        if run_once ==0 and not iteration_n > starting_n:
                            iteration_n += 1
                            iteration_l += 1
                            total_iteration -= iteration_l
                            pass

                        else:
                            print(f"total iteration left is {total_iteration}")
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
                            collection_folder = './Results/%s_collection%d' % (proj_name,layers)
                            if not os.path.exists(collection_folder):
                                os.makedirs(collection_folder)
                            output_folder = './Results/%s_collection%d/intermediate_output%d'%(proj_name,layers,iteration_n)
                            if not os.path.exists(output_folder):
                                os.makedirs(output_folder)

                  
                    
                            if run_once is 0:
                                file_ini = output_folder+'/weights-'+str(epoch_num)+'*'
                                filename = glob.glob(file_ini)
                                if len(filename) == 0:
                                    iteration_n -= 1
                                    output_folder = './Results/%s_collection%d/intermediate_output%d'%(proj_name,layers,iteration_n)
                                    file_ini = output_folder+'/weights-'+str(epoch_num)+'*'
                                    filename = glob.glob(file_ini)
                                print(filename)
                                if os.path.isfile(filename[0]):
                                    model.load_weights(filename[0])
                                else:
                                    print("%s does not exists" % (filename[0]))

                    
                        
                            filepath=output_folder+"/weights-{epoch:02d}-{val_R_squared:.2f}.hdf5"
                            checkpoint = keras.callbacks.ModelCheckpoint(filepath, monitor='val_R_squared', verbose=1, save_best_only=False, save_weights_only=True, mode='auto', period=10)
                            callbacks_list = [earlystop,checkpoint]
                            start = time.time()

                            if run_once is 1:
                                history = model.fit(X_train,Y_train,epochs=300, batch_size=10,callbacks=callbacks_list,validation_split=0.2,verbose=0)
                            else:
                                history = model.fit(X_train,Y_train,epochs=300,batch_size=10,callbacks=callbacks_list,validation_split=0.2,initial_epoch=epoch_num+1,verbose=0)


                            end = time.time()
                            cumulative_time += (end-start)
                            print('it already took %0.2f seconds' % (cumulative_time))
                            scores = model.evaluate(X_test,Y_test,verbose=0)
                            if not os.path.exists("./Results/%s_results%d.txt"%(proj_name,layers)):
                                f = open("./Results/%s_results%d.txt"%(proj_name,layers),"w+")
                            else:
                                f = open("./Results/%s_results%d.txt"%(proj_name,layers),"a+")
                            f.write("For this combination %s, R is %0.2f\r\n" %(parameter_list,scores[1]))
                            if scores[1]>best_R:
                                best_param = parameter_list
                                best_R = scores[1]
                            else:
                                pass
                            f.write("The best_R for now is %0.4f and combination is %s "% (best_R,best_param))
                            print("best R for the combination %s with %d hidden layer is %0.4f" % (best_param,layers ,best_R))
                            run_once = 1

                            iteration_n += 1
                            x ={"layer_number":layers,"starting_n":iteration_n-1,"best_R":best_R,"best_param":best_param,"cumulative_time":cumulative_time}
                            print(x)
                            pr.check_write(x,'%s_latest.json'%(proj_name))
                            print("")

            best_list.append(best_param)
            best_R_list.append(best_R)
            updated_list = {"best_list":best_list,"best_R_list":best_R_list}
            pr.check_write(updated_list,"list_%s.json"%(proj_name)) 
            restart = False
    max_R = best_R_list[0]
    max_param = best_list[0]
    for i in range(len(best_R_list)-1):
        if best_R_list[i+1] > best_R_list[i]:
            max_R = best_R_list[i+1]
            max_param = best_list[i+1]
        else:
            pass
    
    return max_R,max_param


