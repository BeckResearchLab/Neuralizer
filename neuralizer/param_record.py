import glob
import os
import time
import json

def check_write(x,filename):
    with open (filename,'w') as f:
        json.dump(x,f)

def check_read(filename):
    if filename:
        with open(filename,'r') as f:
            y = json.load(f)
    return y

def get_epoch_num(layer_num):
    path_list = glob.glob('**/*.hdf5',recursive = True)
    path = []
    for i in range(len(path_list)):
        if int(path_list[i][18]) == layer_num:
            path.append(path_list[i])
    large_epoch = 0
    l = 0
    for n in range(len(path)):
        a = path[n]
        indices1 = [i for i, c in enumerate(a) if c == '/']
        indices2 = [i for i, c in enumerate(a) if c == '-']
#        print(a)
#       print(indices1)
#      print(indices2)
        number = int(a[39:indices1[2]])
        epoch_num = int(a[indices2[0]+1:indices2[1]])
        if number > l:
            l = number
            large_epoch = epoch_num
        elif number == l:
            if epoch_num > large_epoch:
                large_epoch = epoch_num
            else:
                pass
        else:
            pass

    return large_epoch
