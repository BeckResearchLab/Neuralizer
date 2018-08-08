# Importing necessary package
import pandas as pd
import numpy as np
import os
import sklearn
from sklearn.model_selection import train_test_split
import keras
import keras.backened as K

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


