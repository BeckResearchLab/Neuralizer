# This module helps choose best paramter combinations on samll portion of data
import data_process as dp
import keras
from keras import *
import keras.backend as K
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import random 
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

