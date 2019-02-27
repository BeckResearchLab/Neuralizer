import os
import pandas as pd
import numpy as np
import data_process as dp

def test_read_file():
    X,Y = dp.read_file('test.tsv',["A","B"],"y")
    assert (X == [[0,1],[3,2],[4,3]]),"The data of predictors is loaded wrong"
    assert (Y == [0,1,5]),"data of Y has problems"

def test_data)info():
    data = {"filename":"test.tsv","X_var":["A","B"],"Y_var":"y"}
    X,Y,input_dim,output_dim = dp.data_info(data)
    assert(input_dim == 2),"Dimension of input layer is not correct"
    assert(output_dim == 1), "Dimension of output layer is not correct"
