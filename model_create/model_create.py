import glob
import os
import time
import json
import keras
import keras.backend as K
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split

def make_combo(option1=['tanh','relu','linear'],
                option2=['0.1','0.01','0.001']):
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

class model_build:
    def __init__(self):
        
        
        
        
        
        
class BestModelSearch:
    def __init__(self,model_build_kwargs = None,params):
        self.hidden_layers =params["hidden_layers"]
        self.activation_funcs =params["activation_funcs"]
        self.units = params["units"]

