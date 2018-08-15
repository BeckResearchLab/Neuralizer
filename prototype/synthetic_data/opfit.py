# Importing necessary package
import pandas as pd
import numpy as np
import os
import sklearn
from sklearn.model_selection import train_test_split
import keras
import keras.backened as K
from keras import *

def read_file(filename,X_var,Y_var):
    '''The function takes name of datafile, list of predictors and response'''
    ''' it returns array_like X and Y for data fitting'''
    location = os.path.abspath(filename)
    df = pd.read_csv('%s'%location,sep='\t')
    X = np.array(df[X_var].values)
    Y = np.array(df[Y_var].values)
    return X,Y

def split_train_test(X,Y,test_fraction,random_state):
    '''This function takes X,Y,test_fraction and random_state number'''
    '''It returns training and test sets with desired seperated size'''
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_fraction, random_state=random_state)
    return X_train, X_test, Y_train, Y_test


def R_squared(y_true,y_pred):
    u = K.sum(K.square(y_true - y_pred))
    v = K.sum(K.square(y_true - K.mean(y_true)))
    return (1 - u/(v + K.epsilon()))

def make_combo(option1=['tanh','relu','linear'],option2=['0.1','0.01','0.001']):
    parameter_combo = []
    for i in option1:
        for j in option2:
            combo = []
            combo.append(i)
            combo.append(j)
            parameter_combo.append(combo)
            
    return parameter_combo

def make_pairwise_list(max_depth=2, options=['tanh', 'softmax', 'relu']):
    combinations = []
    for i in range(len(options)**max_depth):
        state = []
        for depth in range(max_depth):
            if depth == 0:
                #print(f"{i:4}:  {options[i // len(options)**(max_depth-1)]}", end='  ')
                state.append(options[i // len(options)**(max_depth-1)])
            elif depth == max_depth - 1:
                #print(f"{options[i % len(options)]}", end='  ')
                state.append(options[i % len(options)])
            else:
                #print(f"{options[i // len(options)**(depth) % len(options)]}", end='  ')
                state.append(options[i // len(options)**(depth) % len(options)])
        #print("")
        combinations.append(state)
    return combinations

def model_search(X_train,Y_train,X_test,Y_test
    input_dim,output_dim,layers,activation_functions=['tanh','softmax','linear']):
    iterations = len(activation_functions)**(layers + 2)
    inner_iterations = len(activation_functions)**layers
    af_combs = make_pairwise_list(max_depth=layers, options=activation_functions)
    print(f'{layers}\t{activation_functions}\t{iterations} iterations required')
    best_R = 0.0
    best_activation = []
    iteration_n = 1
    for n in range(layers):
        best_activation.append('none')
    best_activation = []
    for inner_iteration in range(inner_iterations):
        for activation_in in activation_functions:
            for layer in range(layers):
                inner_list = []
                inner_list.append(af_combs[inner_iteration][layer])
                for activation_out in activation_functions:
                    print(f"running iteration {iteration_n}")
                    parameter_list = []
                    parameter_list.append(activation_in)
                    parameter_list.extend(inner_list)
                    parameter_list.append(activation_out)
                    print(f"create input layer with activation of {activation_in}")
                    
                    model = keras.Sequential()
                    model.add(keras.layers.Dense(10,input_dim = input_dim,activation=activation_in))
                    for i in range(len(inner_list)):
                        print(f"create hidden layer {layer} of type {inner_list[i]}")
                        model.add(keras.layers.Dense(20,activation = inner_list[i]))
                    print(f"create output layer with activation of {activation_out}")
                    model.add(keras.layers.Dense(output_dim,activation=activation_out))
                    model.compile(loss='mean_squared_error', optimizer='adam', 
                                  metrics=[R_squared])
                    history = model.fit(X_train,Y_train,epochs=50, batch_size=10,validation_split=0.33)
                    scores = model.evaluate(X_test,Y_test)
                    print(score)
                    iteration_n += iteration_n 
                    if scores[1]>best_R:
                        best_activation = parameter_list
                        best_R = scores[1]
                    else:
                        pass
        print("")
    print(best_activation)
    print(best_R)
    
    return best_activation

 
def model__multi_search(X_train,Y_train,X_test,Y_test,input_dim,output_dim,layers,
                        activation_functions=['tanh', 'softmax', 'relu'],Dropout=[0.1,0.5,0.8]):
    iterations = (len(Dropout)*len(activation_functions))**(layers+2)
    inner_iterations = (len(Dropout)*len(activation_functions))**layers
    options= make_combo(option1=activation_functions,option2=Dropout)
    af_combs = make_pairwise_list(max_depth=layers, options=options)
    print(f'{layers}\t{activation_functions}\t{iterations} iterations required')
    best_R = 0.0
    best_param = []
    iteration_n = 1
    for n in range(layers):
        best_activation.append('none')
    best_activation = []
    for inner_iteration in range(inner_iterations):
        for option_in in options:
            for layer in range(layers):
                inner_list = []
                inner_list.append(af_combs[inner_iteration])
                for option_out in options:
                    print(f"running iteration {iteration_n}")
                    parameter_list1 = []
                    parameter_list.append(option_in)
                    parameter_list.extend(inner_list)
                    parameter_list.append(option_out)
                    print(f"create input layer with activation of {option_in[0]} and Dropout of {option_in[1]}")
                    
                    model = keras.Sequential()
                    model.add(keras.layers.Dense(10,input_dim = input_dim,activation=option_in[0],
                               Dropout = option_in[1]                 ))
                    for i in range(len(inner_list)):
                        print(f"create hidden layer {layer} of activation {inner_list[i][0]} and Dropout {inner_list[i][1]}")
                        model.add(keras.layers.Dense(20,activation = inner_list[i][0],Dropout = inner_list[i][1]))
                    print(f"create output layer with activation of {option_out[0]} and Dropout of {option_out[1]}")
                    model.add(keras.layers.Dense(output_dim,activation=option_out[0],Dropout=option_out[1]))
                    model.compile(loss='mean_squared_error', optimizer='adam', 
                                  metrics=[R_squared])
                    history = model.fit(X_train,Y_train,epochs=50, batch_size=10,validation_split=0.33)
                    scores = model.evaluate(X_test,Y_test)
                    
                    iteration_n += iteration_n 
                    if scores[1]>best_R:
                        best_param = parameter_list
                        best_R = scores[1]
                    else:
                        pass
        print("")
    print(best_param)
    print(best_R)
    return best_param
    
def layer_search(hidden_layers=[1,3,5],X_train,Y_train,X_test,Y_test
		 input_dim,output_dims
 		 activation_functions,lr):
    best_list1=[]
    best_list2=[]
    for layer_count in hidden_layers:
         best_activation,best_lr = model_multi_search(X_train=X_train,
						      Y_train=Y_train,
				                      X_test=X_test,
						      Y_test=Y_test,
				 		      input_dim=input_dim,
						      output_dims=output_dims,
				 		      layers=layer_count,
						      activation_functions=activationfunctions
						      lr=lr)
            					      
	 best_list1.append(best_activation)
	 best_list2.append(best_lr)    
    
