import os
import pandas as pd
import numpy as np

def read_file(filename, X_var, Y_var):
    """
    The function takes name of datafile, list of predictors and response
    it returns array_like X and Y for data fitting
    """
    location = os.path.abspath('./data/%s/'%filename)
    df = pd.read_csv('%s'%location,sep='\t')
    df.dropna()
    X = np.array(df[X_var].values)
    Y = np.array(df[Y_var].values)
    return X,Y

def data_info(data):
    """
    This function takes in dictionary  with name of datafile, list of predictors and response
    It extracts importatnt information of data for model fitting
    it returns the X,Y,input_dim and output_dim
    """
    filename = data["filename"]
    X_var = data["X_var"]
    Y_var = data["Y_var"]
    X,Y = read_file(filename,X_var,Y_var)
    input_dim = len(X_var)
    output_dim = len(Y_var)
    return X,Y,input_dim,output_dim


            
            
