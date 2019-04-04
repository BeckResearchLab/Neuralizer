from __future__ import absolute_import,division,print_function
import os
import pandas as pd
import numpy as np
import numpy.testing as npt
import data_process as dp

def test_read_file():
    X,Y = dp.read_file('test.tsv',["A","B"],"y")
    npt.assert_equal(X,np.array[[0,1],[3,2],[4,3]])
    npt.assert_equal(Y,np.array[0,1,5])

def test_data_info():
    data = {"filename":"test.tsv","X_var":["A","B"],"Y_var":"y"}
    X,Y,input_dim,output_dim = dp.data_info(data)
    assert input_dim == 2,"Dimension of input layer is not correct"
    assert output_dim == 1 , "Dimension of output layer is not correct"
