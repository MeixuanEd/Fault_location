
import numpy as np
import tensorflow as tf
from sklearn import datasets
from sklearn import svm 
import time
start_time = time.time()
import pandas as pd
import numpy as np
from scipy import *
from numpy import dot, multiply, diag, power
from numpy import pi, exp, sin, cos, cosh, tanh, real, imag
from numpy.linalg import inv, eig, pinv,norm
from scipy.linalg import svd, svdvals
import tensorflow as tf
import os
import matplotlib.pyplot as plt
from io import BytesIO
from functools import partial 
from IPython.display import clear_output, Image, display, HTML  
import scipy.io as sio 
import time  
import re
import warnings
warnings.filterwarnings('ignore')

# parameters 
n_classes=87 
times = 1
buses =68  
def load_data(w,path,name): 
    global times, buses
    import scipy.io as sio 
    PathName = os.path.join(path, name)
    data=sio.loadmat(PathName); 
    dV= data['dV_feature'] 
    Y_ad= data['Y'] 
    train_data = (Y_ad[:,w] @ dV[w,:]).imag.T   
    train_labels = data['y_num'] 
    col, buses = np.shape(train_data)  
    train_x = np.float64(np.reshape(train_data, (int(col/times), buses )))  
    train_y = train_labels.T 
    return train_x, train_y ,col

# load data
def load_all_data(w, fault_type, impe_type):
    global train_data, train_labels,  test_data, test_labels, test_num, eval_data, eval_labels, linedata
    rootPath = '../01_datasets/Iu_feature'
    trainName ='Line_faults_train'   
    testName = 'Line_faults_test' +'_type_' + str(fault_type) + '_'+str(impe_type)
    evalName = 'Line_faults_eval' 
    data = sio.loadmat(os.path.join(rootPath, trainName))
    linedata = data['line']  
    rootPath = '../01_datasets/Iu_feature'
    trainName ='Line_faults_train'   
    testName = 'Line_faults_test' +'_type_' + str(fault_type) + '_'+str(impe_type)
    evalName = 'Line_faults_eval' 
    data = sio.loadmat(os.path.join(rootPath, trainName))
    linedata = data['line']
    train_data, train_labels, train_num = load_data(w,rootPath, trainName)  
    eval_data, eval_labels,eval_num= load_data(w,rootPath, evalName)  
    test_data, test_labels,test_num= load_data(w,rootPath, testName)   

def main():    
    fault_type = 1; # 0--Tp; 1--LG; 2--LLG; 3--LL 
    w= [0,  1,  5,  8, 15, 25, 29, 30, 35, 44, 51, 24,  4, 20, 28, 40, 21, 37, 11, 18]
    total_loss = []
    total_acc = []
    if fault_type == 0:
        totalnum = 2
    else:
        totalnum =5
    for impe_type in range(1, totalnum):  
        load_all_data(w, fault_type, impe_type ) 
        clf = svm.SVC(decision_function_shape = 'ovr')
        clf.fit(train_data, train_labels) 
        dec = clf.decision_function(test_data)
        y_pred = clf.predict(test_data)
        y_pred = np.reshape(y_pred, (test_num,1)) 
        acc = clf.score(test_data,test_labels) 
        total_acc.append(100*acc) 
    print (total_acc)
    print ('The averaged acc of fault ' + '{:d}'.format(fault_type) +  ' is ' +             '{:.2f}'.format(np.mean(total_acc))  ) 
    
if __name__ == '__main__':
    main()



