
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
# parameters

batch_size =50  
n_classes=87 
buses = 68
times =1
rootPath = '../01_datasets/Iu_feature'
trainName ='Line_faults_train' 
data = sio.loadmat(os.path.join(rootPath, trainName))
linedata = data['line'] 

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
    train_x = np.float64(np.reshape(train_data, (int(col/times), buses)))  
    train_y = np.zeros((int(col/times), n_classes))
    for i in range(int(col/times)):
        train_y[i,train_labels[0][i] ] = 1;
    return train_x, train_y ,col
 
def choose_w(thres):
    global buses 
    rootPath = '../01_datasets/Iu_feature'
    trainName ='Line_faults_train'   
    data = sio.loadmat(os.path.join(rootPath, trainName))
    line  = data['line'] 
    all_freq =np.zeros((1,buses))
    for i in range( buses):
        ifreq = size(np.where(line[:,0] == i+1 )) + size(np.where(line[:,1] == i+1))
        all_freq[0][i] = ifreq
    w = [i for i in range(buses) if all_freq[0][i] >thres]  
    return w

# load data
def load_all_data(w, dataname):
    global rootPath, testName  
    test_data, test_labels,test_num= load_data(w,rootPath, dataname)  
    print (np.shape(test_data))
    return test_data, test_labels
 
def load_model(session, save_path):
    global batch_size, n_classes
    """ Loads a saved TF model from a file.Returns:The inputs placehoder and the prediction operation. """
    print("Loading model from file '%s'..." % (save_path))
    meta_file = save_path + ".meta"
    if not os.path.exists(meta_file):
        print("ERROR: Expected .meta file '%s', but could not find it." %         (meta_file))
        sys.exit(1) 
    saver = tf.train.import_meta_graph(meta_file)
    save_path = os.path.join("./", save_path) 
    print ('save_path is', save_path)
    saver.restore(session, save_path)   
    valid_nodes = tf.get_collection_ref("validation_nodes") 
    x = valid_nodes[0]  
    y =valid_nodes[1]
    y_score  = valid_nodes[2] 
    #predict_op=tf.argmax(tf.nn.softmax(pred), 1)   
    #batch_size = int(x.get_shape()[0])  
    # Predict op should also yield integers.
    #predict = tf.cast(predict_op, "int32") 
    # Check the shape of the prediction output.
    #p_shape = predict.get_shape()
    #Commented these out because there could be squeezes in the code earlier 
    return (x, y, y_score ) 

def validate_model(session, val_data, x,  y, y_score  ): 
    """ Validates the model stored in a session.Returns:The overall validation accuracy for the model. """
    global batch_size, n_classes
    print("Validating model...")
    predict_op=tf.argmax(y_score, 1)  
    #y = tf.placeholder(tf.int32, [None, n_classes]) 
    correct = tf.equal(predict_op,tf.argmax(y, 1)) 
# Compute total number of correct answers. 
    acc_rate = tf.reduce_mean(tf.cast(correct, tf.float32)) 

  # Validate the model.
    val_x, val_y  = val_data 
    acc  = session.run(acc_rate, feed_dict={x: val_x, y :val_y      }) 
    correct_results, prob=session.run([correct,y_score],feed_dict={x: val_x, y :val_y    }) 
    return  acc, correct_results,prob


def each_perform(correct_results,eval_labels ):
    label_y=eval_labels#np.argmax(eval_labels,1)
    acc_rate = np.zeros((1,n_classes ))
    for i in range(n_classes):
        location = np.where(label_y == i)  
        print ('location is',location)
        correct = [correct_results[j] for j in location[0]] 
        print (correct)
        if len(correct) > 0:
            acc_rate[0][i] = 100*np.mean(correct)  
        #print ('acc is',100*acc_rate[i])
    return acc_rate
def reset():
    tf.reset_default_graph()  
    
def main():
    fault_type =3# 0--Tp; 1--LG; 2--LLG; 3--LL
    impe_num = 4;   # 1~4 fault impedance increases 

    model_name = './best_NN_30_ratio/NN_locate_30'#'NN_locate_4-layer_30'#'NN_locate_half'#'NN_locate_full' NN_locate_7
    thres =4
    w = choose_w(  thres)
    w = np.r_[w, 23,42,61,51,57,6,37,58,45,26,10,33,4,41,8] 
    accuracy = np.zeros((impe_num,1))
    total_score=[] 
    if fault_type == 0:
        totalnum = 2
    else:
        totalnum = 5
    accuracy = np.zeros((totalnum-1,1))   
    for impe_type in range(1,totalnum): 
        testName = 'Line_faults_test' +'_type_' + str(fault_type) + '_'+str(impe_type ) 
        val_data = load_all_data(w, testName)
        with tf.Session() as session:   
            best_model =  model_name  
            x, y, y_score  = load_model(session, best_model) 
            accuracy[impe_type-1]  ,correct_results ,y_prob  = validate_model(session, val_data, x,y, y_score) 
            print ("Overall validation accuracy is %f \n" %(100*accuracy[impe_type-1] ))   
            total_score.append(y_prob)
        session.close()    
        reset()

    print ('Mean acc is {:.2f}'.format(100*np.mean(accuracy))) 
if __name__ == '__main__':
    main()    


